from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os
import traceback
from datetime import timedelta
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- 1. ENVIRONMENT & CLIENT INITIALIZATION (From Step 1) ---
load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    print("🛑 FATAL: Alpaca API keys not found in environment.")
    sys.exit(1)

# Initialize all Alpaca clients explicitly
trade_client = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=True)
option_historical_data_client = OptionHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
stock_data_client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

# --- 2. QOPS INTERNAL IMPORTS ---
from qops.data.alpaca_fetch import calculate_greeks, get_option_chain_snapshot
from qops.strategy.selector import CandidateSelector

# --- 3. CONFIGURATION & THRESHOLDS ---
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SG_CSV = REPO_ROOT / "data" / "SG" / "INTC_history_table_2026-03-07.csv"
DEFAULT_MODEL_DIR = REPO_ROOT / "models"

# Step 1 Timezone & Financials
NY_TZ = ZoneInfo("America/New_York")
RISK_FREE_RATE = 0.01  # Aligned with Step 1 defaults
BUFFER_PCT = 0.05

SYMBOL = "INTC"
STRATEGY_MODE = "CALL_SPREAD_CREDIT"
ENTRY_HOUR = 9
ENTRY_MINUTE = 45
EXIT_HOUR = 15
EXIT_MINUTE = 0

# Widened Delta & Width Thresholds for ML Variance
SHORT_DELTA_RANGE = (0.10, 0.50)
LONG_DELTA_RANGE = (0.01, 0.25)
SPREAD_WIDTH = (0.5, 10.0)

def _sanitize_SG_df(raw: pd.DataFrame) -> pd.DataFrame:
    sg_df = raw.copy()
    sg_df.columns = sg_df.columns.str.strip().str.replace("\xa0", " ")

    for col in sg_df.columns:
        if sg_df[col].dtype == "object" and col != "Trade Date" and "Exp" not in col:
            try:
                sg_df[col] = sg_df[col].str.replace("'", "").str.replace(",", "").astype(float)
            except ValueError:
                pass

    sg_df['Trade Date'] = pd.to_datetime(sg_df['Trade Date']).dt.date
    sg_df.set_index('Trade Date', inplace=True)
    return sg_df

def run_training_pipeline(csv_path: Path, model_dir: Path, symbol: str, strategy_mode: str, 
                          entry_h: int, entry_m: int, exit_h: int, exit_m: int):
    print(f"🚀 Starting ML Training Pipeline for {symbol}")
    
    if not csv_path.exists():
        print(f"🛑 FATAL: Could not find SG CSV at {csv_path}")
        sys.exit(1)

    raw_sg = pd.read_csv(csv_path)
    sg_df = _sanitize_SG_df(raw_sg)
    training_dates = sg_df.index.unique().tolist()

    master_training_data = []

    for current_date in training_dates:
        try:
            dt = pd.to_datetime(current_date).tz_localize(NY_TZ)
            entry_time = dt.replace(hour=entry_h, minute=entry_m)
            
            sg_row = sg_df.loc[current_date]
            spot = float(sg_row.get('Previous Close', 45.00)) 

            # 1. Fetch Options Snapshot (Using Step 1 buffer logic)
            chain_df = get_option_chain_snapshot(symbol, entry_time, spot)
            if chain_df.empty:
                continue

            # 2. Calculate Greeks (Passing our defined Risk-Free Rate)
            chain_with_greeks = calculate_greeks(chain_df, spot, risk_free_rate=RISK_FREE_RATE)
            
            # 3. Select Candidates (Passing our widened global thresholds)
            selector = CandidateSelector(chain_with_greeks, spot)
            candidates = selector.get_spread_candidates(
                strategy_mode, 
                spread_width=SPREAD_WIDTH,
                short_delta_range=SHORT_DELTA_RANGE,
                long_delta_range=LONG_DELTA_RANGE
            )
            
            if candidates.empty:
                continue
                
            top_trade = candidates.iloc[0]

            # --- 4. TRUE LABEL GENERATION (WIN/LOSS) ---
            exit_time = dt.replace(hour=exit_h, minute=exit_m)
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=exit_time - timedelta(minutes=10),
                end=exit_time
            )
            # Utilizing the explicitly defined stock_client
            stock_bars = stock_client.get_stock_bars(req).df
            
            if stock_bars.empty:
                continue
                
            close_price = stock_bars.iloc[-1]['close']
            
            # Call Credit Spread: We win if INTC closes BELOW the short call strike
            is_winner = 1 if close_price < top_trade['short_strike'] else 0

            # --- 5. FEATURE ENGINEERING ---
            high_vol = sg_row.get('High Vol Point', spot)
            low_vol = sg_row.get('Low Vol Point', spot)
            range_width = high_vol - low_vol
            
            master_training_data.append({
                'net_delta': top_trade['net_delta'],
                'alpha': top_trade['premium'] / (top_trade['max_loss'] + top_trade['premium']),
                'spot_in_vol_range': (spot - low_vol) / range_width if range_width > 0 else 0.5,
                'dist_to_high_vol': (top_trade['short_strike'] - high_vol) / high_vol,
                'dist_to_low_vol': (top_trade['short_strike'] - low_vol) / low_vol,
                'is_winner': is_winner
            })

        except Exception as e:
            print(f"❌ Skipping {current_date} due to error: {e}")
            continue

    # --- 6. MODEL TRAINING ---
    train_df = pd.DataFrame(master_training_data)
    
    if train_df.empty:
        print("🛑 FATAL: No training data generated. Exiting.")
        sys.exit(1)

    print(f"\n✅ Generated {len(train_df)} valid training rows.")

    features = ['net_delta', 'alpha', 'spot_in_vol_range', 'dist_to_high_vol', 'dist_to_low_vol']
    X = train_df[features].fillna(0)
    y = train_df['is_winner']

    if len(y.unique()) < 2:
        print(f"🛑 FATAL: Dataset contains only one class. Try adjusting delta thresholds.")
        sys.exit(1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(class_weight='balanced')
    model.fit(X_scaled, y)

    print("\n🚀 Model Trained Successfully! Learned Weights:")
    for feat, coef in zip(features, model.coef_[0]):
        print(f"   {feat}: {coef:.4f}")

    # Save Artifacts
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "synthetic_logistic_model.pkl")
    joblib.dump(scaler, model_dir / "synthetic_scaler.pkl")
    print(f"\n✅ ML Artifacts saved to {model_dir}/")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train SyntheticMLScorer model artifacts.")
    p.add_argument("--SG-csv", type=Path, default=DEFAULT_SG_CSV, help="Path to SG history table CSV.")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory to write model/scaler artifacts.")
    p.add_argument("--symbol", type=str, default=SYMBOL, help="Underlying symbol (e.g. INTC).")
    p.add_argument("--strategy-mode", type=str, default=STRATEGY_MODE)
    p.add_argument("--entry-hour", type=int, default=ENTRY_HOUR)
    p.add_argument("--entry-minute", type=int, default=ENTRY_MINUTE)
    p.add_argument("--exit-hour", type=int, default=EXIT_HOUR)
    p.add_argument("--exit-minute", type=int, default=EXIT_MINUTE)
    return p

def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    run_training_pipeline(
        args.SG_csv, args.model_dir, symbol=args.symbol, strategy_mode=args.strategy_mode,
        entry_h=args.entry_hour, entry_m=args.entry_minute, exit_h=args.exit_hour, exit_m=args.exit_minute
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
