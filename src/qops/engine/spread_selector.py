def find_short_and_long_puts(
    df,
    risk_free_rate,
    short_put_delta_range,
    long_put_delta_range,
    spread_width=(2,4)
):

    short_put = None
    long_put = None

    for ts in sorted(df["timestamp"].unique()):

        snapshot = df[df["timestamp"] == ts]

        for _, row in snapshot.iterrows():

            delta = row["delta"]
            strike = row["strike_price"]

            if short_put_delta_range[0] <= delta <= short_put_delta_range[1]:
                short_put = row

            if long_put_delta_range[0] <= delta <= long_put_delta_range[1]:
                long_put = row

            if short_put is not None and long_put is not None:

                width = short_put["strike_price"] - long_put["strike_price"]

                if spread_width[0] <= width <= spread_width[1]:
                    return short_put, long_put

                short_put = None
                long_put = None

    raise ValueError("No valid spread found")
