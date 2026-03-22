def run_iterative_backtest(
    df,
    risk_free_rate,
    short_delta_range,
    long_delta_range,
    spread_width,
    max_iterations=50
):

    results = []

    start_ts = df["timestamp"].min()

    iteration = 1

    while iteration <= max_iterations:

        subset = df[df["timestamp"] >= start_ts]

        if subset.empty:
            break

        try:

            short_put, long_put = find_short_and_long_puts(
                subset,
                risk_free_rate,
                short_delta_range,
                long_delta_range,
                spread_width
            )

        except ValueError:
            break

        status, pnl, exit_ts = simulate_bull_put_trade(
            subset,
            short_put,
            long_put,
            risk_free_rate,
            delta_stop_mult=2,
            profit_target=0.5
        )

        results.append({
            "status": status,
            "pnl": pnl,
            "entry_time": short_put["timestamp"],
            "exit_time": exit_ts,
            "short": short_put["option_symbol"],
            "long": long_put["option_symbol"]
        })

        start_ts = exit_ts + pd.Timedelta(minutes=1)
        iteration += 1

    return results
