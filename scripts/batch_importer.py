"""Batch import and analysis of TradingView backtest exports from backtest_exports/."""

import json
from pathlib import Path

import pandas as pd

# TradingView export uses row-based key-value pairs: row label in col 0, values in col 1 (All USD) or col 2 (All %).
# Maps (section_contains, row_label) -> (output_key, value_column_index).
BACKTEST_EXPORT_STAT_MAP = {
    ("Performance", "Net profit"): ("Net profit", 1),
    ("Performance", "Initial capital"): ("Initial capital", 1),
    ("Trades analysis", "Percent profitable"): ("Win Rate %", 2),
    ("Trades analysis", "Avg P&L"): ("Avg Trade %", 2),
    ("Risk-adjusted", "Profit factor"): ("Profit factor", 1),
    ("Risk-adjusted", "Sharpe ratio"): ("Sharpe ratio", 1),
}

OUTPUT_DIR = Path("data/backtests")


def _parse_tv_export(csv_path: Path) -> tuple[dict, float, pd.DataFrame | None]:
    """Parse a single TV backtest CSV. Returns (stats_dict, initial_capital, trades_df or None)."""
    raw = pd.read_csv(csv_path, header=None)
    col0 = raw.iloc[:, 0].astype(str).str.strip()
    section_starts = col0[col0.str.contains("Table 1", na=False)].index.tolist()

    def get_val(section_substr: str, row_label: str, value_col: int):
        for i in section_starts:
            if section_substr not in col0.iloc[i]:
                continue
            start, end = i + 2, section_starts[section_starts.index(i) + 1] if section_starts.index(i) + 1 < len(section_starts) else len(raw)
            for j in range(start, end):
                if col0.iloc[j] == row_label:
                    val = raw.iloc[j, value_col]
                    if pd.isna(val) or val == "":
                        return 0.0
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return 0.0
        return 0.0

    stats = {}
    for (section, label), (out_key, col_idx) in BACKTEST_EXPORT_STAT_MAP.items():
        stats[out_key] = get_val(section, label, col_idx)
    initial_capital = stats.get("Initial capital", 100000.0)
    stats.pop("Initial capital", None)

    trades_df = None
    for i in section_starts:
        if "List of trades" not in col0.iloc[i]:
            continue
        header_row = i + 1
        data_start = i + 2
        next_section = next((s for s in section_starts if s > i), len(raw))
        n_rows = next_section - data_start
        if n_rows < 1:
            break
        header = raw.iloc[header_row].astype(str).str.strip().tolist()
        data = raw.iloc[data_start:next_section]
        n_cols = min(len(header), data.shape[1])
        trades_df = pd.DataFrame(data.iloc[:, :n_cols].values, columns=header[:n_cols])
        break

    return stats, initial_capital, trades_df


def _equity_curve_from_trades(trades_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame | None:
    """Build equity curve from List of trades (Date and time + Cumulative P&L USD)."""
    if trades_df is None or trades_df.empty:
        return None
    date_col = next((c for c in trades_df.columns if c.strip() == "Date and time"), None)
    cum_col = next((c for c in trades_df.columns if c.strip() == "Cumulative P&L USD"), None)
    type_col = next((c for c in trades_df.columns if c.strip() == "Type"), None)
    if not date_col or not cum_col:
        return None
    subset = trades_df[[date_col, cum_col]].copy()
    if type_col and subset.shape[0] > 0 and "Exit" in str(trades_df[type_col].iloc[0]):
        subset = trades_df.loc[trades_df[type_col].astype(str).str.contains("Exit", na=False), [date_col, cum_col]].copy()
    subset.columns = ["date", "cumulative_pnl"]
    subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
    subset = subset.dropna(subset=["date"])
    subset["cumulative_pnl"] = pd.to_numeric(subset["cumulative_pnl"], errors="coerce").fillna(0)
    subset = subset.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    subset["equity"] = initial_capital + subset["cumulative_pnl"]
    return subset[["date", "equity", "cumulative_pnl"]].reset_index(drop=True)


def analyze_orb_results(folder_name: str = "backtest_exports") -> dict | None:
    """Scan backtest_exports for TV CSVs, parse stats and equity curves, return comparison and artifacts."""
    results_dir = Path.cwd() / folder_name
    results_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("--- Strategy Analysis Active ---")
    print(f"Scanning directory: {results_dir.resolve()}")
    print("--------------------------------")

    report_data = []
    equity_curves = {}

    files = sorted(results_dir.glob("*.csv"))
    if not files:
        print("Status: Empty. Please drop your TV exports in the folder above.")
        return None

    for file_path in files:
        parts = file_path.stem.split("_")
        if len(parts) < 3:
            print(f"Skipping format error: {file_path.name}")
            continue
        ticker, timeframe, version = parts[0], parts[1], parts[2]
        try:
            stats, initial_capital, trades_df = _parse_tv_export(file_path)
            row = {"Ticker": ticker, "TF": timeframe, "Ver": version, **stats}
            report_data.append(row)
            curve = _equity_curve_from_trades(trades_df, initial_capital)
            if curve is not None and not curve.empty:
                key = f"{ticker}_{timeframe}_{version}"
                equity_curves[key] = curve
        except Exception as e:
            print(f"Read error on {file_path.name}: {e}")

    if not report_data:
        return None

    summary_df = pd.DataFrame(report_data)
    comparison = summary_df.pivot_table(
        index=["Ticker", "TF"],
        columns="Ver",
        values=["Net profit", "Profit factor", "Sharpe ratio", "Avg Trade %"],
    )

    summary_payload = {
        "summary": report_data,
        "equity_curve_keys": list(equity_curves.keys()),
    }
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Wrote {summary_path}")

    for key, curve in equity_curves.items():
        out_path = OUTPUT_DIR / f"equity_curve_{key}.csv"
        curve.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    return {
        "comparison": comparison,
        "summary": report_data,
        "summary_path": str(summary_path),
        "equity_curves": equity_curves,
    }


if __name__ == "__main__":
    results = analyze_orb_results()
    if results is not None:
        print("\n--- Final A/B Comparison ---")
        print(results["comparison"])
        print("\n--- Equity curve keys ---")
        print(results["equity_curve_keys"])