import argparse
from pathlib import Path

from qops.engine.backtester import BacktestConfig, run_backtest


def cmd_backtest_run(args):
    cfg = BacktestConfig(
        candidates_csv=args.candidates,
        out_dir=args.out_dir,
        entry_time_et=args.entry,
        exit_time_et=args.exit,
        strike_min_mult=args.strike_min,
        strike_max_mult=args.strike_max,
        price_cap=args.price_cap,
        min_entry_minute_volume=args.min_entry_vol,
        min_prev_day_volume=args.min_prev_vol,
        prev_day_top_n=args.top_n,
        slippage_pct=args.slippage,
        mode=args.mode,
        intc_expiries_csv=args.expiries,
    )

    trades, summary = run_backtest(cfg)

    print("\nBacktest complete")
    print("------------------")
    print(summary)


def build_parser():

    parser = argparse.ArgumentParser(prog="qops")
    sub = parser.add_subparsers(dest="command")

    # BACKTEST COMMAND
    backtest = sub.add_parser("backtest")
    bt_sub = backtest.add_subparsers(dest="bt_command")

    run = bt_sub.add_parser("run")

    run.add_argument("--mode", default="candidates")
    run.add_argument("--candidates", default="data/candidates_v3.csv")
    run.add_argument("--expiries", default="data/intc_expiries.csv")

    run.add_argument("--entry", default="09:45")
    run.add_argument("--exit", default="15:00")

    run.add_argument("--strike-min", type=float, default=0.98)
    run.add_argument("--strike-max", type=float, default=1.12)

    run.add_argument("--price-cap", type=float, default=3.50)

    run.add_argument("--min-entry-vol", type=int, default=10)
    run.add_argument("--min-prev-vol", type=int, default=300)
    run.add_argument("--top-n", type=int, default=5)

    run.add_argument("--slippage", type=float, default=0.03)

    run.add_argument("--out-dir", default="data/backtests")

    run.set_defaults(func=cmd_backtest_run)

    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
