import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# Alpaca Imports
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionBarsRequest
from alpaca.data.timeframe import TimeFrame

# QOPS Internal Modules (Assuming these exist in your repo)
from qops.data.alpaca_fetch import get_option_chain_snapshot, calculate_greeks
from qops.strategy.selector import CandidateSelector
from qops.strategy.scorer import SyntheticMLScorer

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

stock_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
option_client = OptionHistoricalDataClient(API_KEY, SECRET_KEY)

SYMBOL = "INTC"
STRATEGY_MODE = "CALL_SPREAD_CREDIT"
START_CAPITAL = 10000.0

# --- 2. INTRADAY RISK EVALUATOR ---
def evaluate_intraday_risk(intraday_options_data: pd.DataFrame, entry_trade: dict, 
                           target_stop_loss_pct: float = 2.0, take_profit_pct: float = 0.5, 
                           delta_stop_thres: float = 0.50) -> dict:
    """Steps through 1-minute historical bars to see if stops were hit before 15:00."""
    initial_credit = entry_trade['premium']
    short_sym = entry_trade['short_symbol']
    long_sym = entry_trade['long_symbol']
    
    exit_price = None
    exit_time = None
    exit_reason = "TIME_EXPIRATION"
    
    if intraday_options_data.empty:
        return {"exit_time": None, "exit_price": 0, "exit_reason": "NO_DATA", "pnl_$": 0}

    for ts, group in intraday_options_data.groupby('timestamp'):
        short_row = group[group['symbol'] == short_sym]
        long_row = group[group['symbol'] == long_sym]
        
        if short_row.empty or long_row.empty:
            continue
            
        short_ask = short_row.iloc[0].get('ask', 0)
        long_bid = long_row.iloc[0].get('bid', 0)
        if short_ask == 0 or long_bid == 0: continue
        
        current_short_delta = abs(short_row.iloc[0].get('delta', 0))
        current_debit = short_ask - long_bid 
        
        # 1. Delta Stop
        if current_short_delta >= delta_stop_thres:
            exit_price, exit_time, exit_reason = current_debit, ts, "DELTA_STOP"
            break
        # 2. Financial Stop
        if current_debit >= (initial_credit * target_stop_loss_pct):
            exit_price, exit_time, exit_reason = current_debit, ts, "MAX_LOSS_STOP"
            break
        # 3. Take Profit
        if current_debit <= (initial_credit * (1 - take_profit_pct)):
            exit_price, exit_time, exit_reason = current_debit, ts, "TAKE_PROFIT"
            break

    # 15:00 Expiry Enforcement
    if exit_price is None:
        last_ts = intraday_options_data['timestamp'].max()
        last_group = intraday_options_data[intraday_options_data['timestamp'] == last_ts]
        
        try:
            s_ask = last_group[last_group['symbol'] == short_sym]['ask'].values[0]
            l_bid = last_group[last_group['symbol'] == long_sym]['bid'].values[0]
            exit_price, exit_time = (s_ask - l_bid), last_ts
        except IndexError:
            exit_price = initial_credit # Fallback if missing closing quote
            
    return {
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl_$": (initial_credit - exit_price) * 100
    }

# --- 3. THE MAIN BACKTEST ENGINE ---
def run_backtest():
    print(f"🚀 Starting Backtest: {SYMBOL} | Strategy: {STRATEGY_MODE}")
    
    # Load SG Data
    # --- LOAD AND SANITIZE SG DATA ---
    sg_df = pd.read_csv("data/SG/INTC_history_table_2026-03-07.csv")
    
    # 1. Strip hidden characters and whitespace from column headers
    sg_df.columns = sg_df.columns.str.strip().str.replace('\xa0', ' ')
    
    # 2. Fix SG's leading single quotes on negative numbers & remove commas
    for col in sg_df.columns:
        if sg_df[col].dtype == 'object' and col != 'Trade Date' and 'Exp' not in col:
            try:
                sg_df[col] = sg_df[col].str.replace("'", "").str.replace(",", "").astype(float)
            except ValueError:
                pass # Ignore strings that are actually dates (like Top Gamma Exp)

    # 3. Standardize the Date Index
    sg_df['Trade Date'] = pd.to_datetime(sg_df['Trade Date']).dt.date
    sg_df.set_index('Trade Date', inplace=True)
    # ----------------------------------------
    trading_days = sg_df.index.unique().tolist()
    
    scorer = SyntheticMLScorer()
    capital = START_CAPITAL
    peak_capital = capital
    max_drawdown = 0.0
    backtest_results = []

    for current_date in trading_days:
        try:
            dt = pd.to_datetime(current_date).tz_localize("America/New_York")
            entry_time = dt.replace(hour=9, minute=45)
            exit_time = dt.replace(hour=15, minute=0)
            
            # Fetch Spot & SG Row
            sg_row = sg_df.loc[current_date]
            spot = float(sg_row.get('Previous Close', 45.00)) # Simplified for loop speed
            
            # 1. Get Chain & Score Candidates
            chain_df = get_option_chain_snapshot(SYMBOL, entry_time)
            if chain_df.empty: continue
            
            chain_with_greeks = calculate_greeks(chain_df, spot)
            selector = CandidateSelector(chain_with_greeks, spot)
            candidates = selector.get_spread_candidates(STRATEGY_MODE)
            
            if candidates.empty: continue
            
            scored = scorer.predict_win_probability(candidates, sg_row, spot)
            top_trade = scored.iloc[0]
            
            # 2. Fetch Intraday 1-Min Bars for the selected legs
            req = OptionBarsRequest(
                symbol_or_symbols=[top_trade['short_symbol'], top_trade['long_symbol']],
                timeframe=TimeFrame.Minute,
                start=entry_time,
                end=exit_time
            )
            intraday_bars = option_client.get_option_bars(req).df.reset_index()
            
            # 3. Evaluate Risk & PnL
            result = evaluate_intraday_risk(intraday_bars, top_trade)
            
            trade_pnl = result['pnl_$']
            capital += trade_pnl
            
            # Drawdown Math
            if capital > peak_capital: peak_capital = capital
            current_drawdown = (peak_capital - capital) / peak_capital
            if current_drawdown > max_drawdown: max_drawdown = current_drawdown
            
            backtest_results.append({
                'date': current_date, 'spot_entry': spot, 'ml_score': top_trade['ml_win_prob'],
                'pnl_$': trade_pnl, 'equity': capital, 'drawdown_pct': current_drawdown * 100,
                'exit_reason': result['exit_reason']
            })
            
            print(f"[{current_date}] PnL: ${trade_pnl:+.2f} | Reason: {result['exit_reason']}")

        except Exception as e:
            print(f"[{current_date}] Error: {e}")

    # --- 4. REPORTING & PLOTTING ---
    df = pd.DataFrame(backtest_results)
    if df.empty:
        print("No trades executed.")
        return

    print("=" * 50)
    print("📊 BACKTESTING SUMMARY")
    print("=" * 50)
    print(f"Starting Capital: ${START_CAPITAL:.2f}")
    print(f"Final Equity:     ${capital:.2f}")
    print(f"Total Return:     {((capital - START_CAPITAL) / START_CAPITAL) * 100:.2f}%")
    print(f"Max Drawdown:     -{max_drawdown * 100:.2f}%")
    print(f"Win Rate:         {(df['pnl_$'] > 0).mean() * 100:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0A0A0A")
    ax.set_facecolor("#0A0A0A")
    ax.plot(pd.to_datetime(df['date']), df['equity'], color="#3498DB", linewidth=2)
    ax.set_title(f"{SYMBOL} {STRATEGY_MODE} | Equity Curve", color="white")
    ax.tick_params(colors="white")
    ax.grid(True, linestyle=":", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("data/backtest_equity_curve.png", facecolor="#0A0A0A")
    print("📈 Chart saved to data/backtest_equity_curve.png")

if __name__ == "__main__":
    run_backtest()