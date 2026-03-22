def simulate_bull_put_trade(
    df,
    short_put,
    long_put,
    risk_free_rate,
    delta_stop_mult,
    profit_target
):

    entry_ts = short_put["timestamp"]
    expiry = short_put["expiration_date"]

    short_symbol = short_put["option_symbol"]
    long_symbol = long_put["option_symbol"]

    short_price = short_put["option_price"]
    long_price = long_put["option_price"]

    initial_credit = short_price - long_price
    initial_delta = short_put["delta"] - long_put["delta"]

    delta_stop = initial_delta * delta_stop_mult
    target_price = initial_credit * profit_target

    timestamps = sorted(df[df["timestamp"] > entry_ts]["timestamp"].unique())

    for ts in timestamps:

        snap = df[df["timestamp"] == ts]

        s = snap[snap["option_symbol"] == short_symbol]
        l = snap[snap["option_symbol"] == long_symbol]

        if s.empty or l.empty:
            continue

        short_bid = s.iloc[0]["bid"]
        long_ask = l.iloc[0]["ask"]

        spread_price = short_bid - long_ask

        pnl = (initial_credit - spread_price) * 100

        if spread_price <= target_price:
            return "profit", pnl, ts

        if abs(initial_delta) >= abs(delta_stop):
            return "delta_stop", pnl, ts

    return "expired", initial_credit*100, timestamps[-1]
