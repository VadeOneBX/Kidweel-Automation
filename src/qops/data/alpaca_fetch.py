from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from dotenv import load_dotenv
from scipy.optimize import brentq
from scipy.stats import norm


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@lru_cache(maxsize=1)
def _option_client() -> OptionHistoricalDataClient:
    """
    Lazily construct the Alpaca client so importing this module doesn't
    immediately depend on env vars (important for unit tests / offline code paths).
    """
    # Dev-friendly: load dotenv if present; no-op if not.
    load_dotenv()
    return OptionHistoricalDataClient(_require_env("APCA_API_KEY_ID"), _require_env("APCA_API_SECRET_KEY"))

# --- 1. DATA SANITIZATION & FILTERING ---

def parse_opra_symbol(opra_string: str):
    """Extracts contract details and Expiration Date from OPRA format."""
    match = re.match(r"^([A-Za-z]+)(\d{6})([CP])(\d{8})$", opra_string)
    if not match: return None, None, None
    
    expiry_str = match.group(2)
    opt_type = match.group(3)
    strike = int(match.group(4)) / 1000.0
    
    # Convert YYMMDD to a real date object
    expiry_date = datetime.strptime(expiry_str, "%y%m%d").date()
    return opt_type, strike, expiry_date

def calculate_strike_price_range(spot_price: float, buffer_pct: float = 0.05) -> Tuple[float, float]:
    """Calculates boundaries to prevent fetching thousands of useless deep OTM options."""
    min_strike = spot_price * (1 - buffer_pct)
    max_strike = spot_price * (1 + buffer_pct)
    return min_strike, max_strike

def get_option_chain_snapshot(symbol: str, snapshot_time, spot_price: float = None) -> pd.DataFrame:
    """Pulls the option chain snapshot, bounding strikes and filtering DTE < 7."""
    req = OptionChainRequest(underlying_symbol=symbol, snapshot_time=snapshot_time)
    chain = option_client.get_option_chain(req)
    
    min_strike, max_strike = 0, float('inf')
    if spot_price:
        min_strike, max_strike = calculate_strike_price_range(spot_price)
        
    snapshot_date = snapshot_time.date()
    data = []
    
    for contract, snapshot in chain.items():
        opt_type, strike, expiry_date = parse_opra_symbol(contract)
        if not opt_type: continue
        
        # --- THE NEW DTE FILTER ---
        dte = (expiry_date - snapshot_date).days
        if dte < 0 or dte >= 7: 
            continue # Throw away anything expiring in 7+ days or already expired
            
        if spot_price and (strike < min_strike or strike > max_strike):
            continue
            
        if snapshot.latest_quote:
            data.append({
                'symbol': contract,
                'strike': strike,
                'type': opt_type,
                'expiry': expiry_date,
                'dte': dte, # Save the exact DTE for the Greeks math
                'bid': float(snapshot.latest_quote.bid_price),
                'ask': float(snapshot.latest_quote.ask_price),
                'volume': 0,
                'open_interest': 0,
            })
    
    df = pd.DataFrame(data).dropna(subset=['bid', 'ask'])
    if not df.empty:
        df['mid_price'] = (df['bid'] + df['ask']) / 2.0
    return df

# --- 2. ROBUST GREEKS MATH (From Alpaca Guide Step 4) ---

def calculate_implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
) -> float:
    """Robust IV calculation using Scipy Brentq to prevent 0DTE math explosions."""
    if T <= 1e-6:
        return 1e-6  # Expired/Expiring

    def bs_price(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type.lower() == "c":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def objective_function(sigma):
        return bs_price(sigma) - option_price

    try:
        # Search for IV between 1% and 300%
        return brentq(objective_function, 1e-4, 3.0)
    except ValueError:
        return 1e-6  # If solver fails, return near-zero rather than crashing

def calculate_greeks(df: pd.DataFrame, spot: float, risk_free_rate=0.05) -> pd.DataFrame:
    """Calculates IV and Delta dynamically based on exact DTE."""
    df = df.copy()
    
    def get_greeks(row):
        flag = 'c' if row['type'].upper() == 'C' else 'p'
        
        # Dynamically calculate T (Time in years). Max prevents divide-by-zero on 0DTE.
        dynamic_T = max(row['dte'] / 365.0, 1e-6)
        
        if dynamic_T == 1e-6:
            if flag == 'p':
                d = -1.0 if spot < row['strike'] else 0.0
            else:
                d = 1.0 if spot > row['strike'] else 0.0
            return pd.Series({'iv': 1e-6, 'delta': d})

        iv = calculate_implied_volatility(row['mid_price'], spot, row['strike'], dynamic_T, risk_free_rate, flag)
        
        if iv <= 1e-6:
            return pd.Series({'iv': np.nan, 'delta': np.nan})
            
        d1 = (np.log(spot / row['strike']) + (risk_free_rate + 0.5 * iv**2) * dynamic_T) / (iv * np.sqrt(dynamic_T))
        d = norm.cdf(d1) if flag == 'c' else norm.cdf(d1) - 1
        
        return pd.Series({'iv': iv, 'delta': d})
            
    greeks_df = df.apply(get_greeks, axis=1)
    return pd.concat([df, greeks_df], axis=1)

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch an Alpaca option chain snapshot to CSV.")
    p.add_argument("--symbol", type=str, required=True, help="Underlying symbol (e.g. INTC).")
    p.add_argument(
        "--snapshot-time",
        type=str,
        required=True,
        help="Snapshot timestamp as ISO-8601 (e.g. 2026-03-07T09:45:00-05:00).",
    )
    p.add_argument("--spot-price", type=float, default=None, help="Spot price for strike-bounding filter.")
    p.add_argument("--out", type=str, default=None, help="Write CSV to this path (optional).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        snapshot_dt = datetime.fromisoformat(args.snapshot_time)
    except ValueError as e:
        raise SystemExit(f"Invalid --snapshot-time: {e}")

    df = get_option_chain_snapshot(args.symbol, snapshot_dt, args.spot_price)
    if args.out:
        pd.DataFrame(df).to_csv(args.out, index=False)
        print(f"Wrote {len(df)} rows to {args.out}")
    else:
        # Print a small preview; avoid dumping thousands of lines.
        with pd.option_context("display.max_rows", 20, "display.width", 120):
            print(df.head(20).to_string(index=False))
            print(f"\nRows: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())