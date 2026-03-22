import pandas as pd


class CandidateSelector:
    def __init__(self, chain_with_greeks: pd.DataFrame, spot_price: float):
        self.chain = chain_with_greeks.dropna(subset=["delta"]).copy()
        self.spot = float(spot_price)

    def get_spread_candidates(
        self,
        spread_type: str,
        spread_width: tuple = (0.5, 10.0),
        short_delta_range: tuple = (0.10, 0.50),
        long_delta_range: tuple = (0.01, 0.25),
        min_dte: int = 1,
        max_dte: int = 13,
        max_strike_dist_pct: float = 0.10,
    ) -> pd.DataFrame:
        """
        Build vertical spread candidates from an option chain with greeks.

        Parameters
        ----------
        spread_type : str
            Examples: "CALL_SPREAD_CREDIT", "PUT_SPREAD_CREDIT", "CALL_SPREAD_DEBIT"
        spread_width : tuple
            Allowed min/max width in strike dollars.
        short_delta_range : tuple
            Allowed absolute delta range for short leg.
        long_delta_range : tuple
            Allowed absolute delta range for long leg.
        min_dte / max_dte : int
            Expiration window to consider.
        max_strike_dist_pct : float
            Filter out contracts whose strike is too far from spot.
        """
        if self.chain.empty:
            return pd.DataFrame()

        is_call = "CALL" in spread_type.upper()
        is_credit = "CREDIT" in spread_type.upper()
        opt_type = "C" if is_call else "P"

        # Base filter: type + DTE window
        df = self.chain[
            (self.chain["type"] == opt_type) &
            (self.chain["dte"] >= min_dte) &
            (self.chain["dte"] <= max_dte)
        ].copy()

        if df.empty:
            return pd.DataFrame()

        # Strike proximity filter: keep only contracts reasonably near spot
        df["strike_dist_pct"] = (df["strike"] - self.spot).abs() / self.spot
        df = df[df["strike_dist_pct"] <= max_strike_dist_pct].copy()

        if df.empty:
            return pd.DataFrame()

        # Ensure expiry is sortable/groupable
        if "expiry" in df.columns:
            df = df.sort_values(["expiry", "strike"])
        else:
            df = df.sort_values(["strike"])

        pairs = []

        # Build spreads within the same expiry only
        expiry_groups = df.groupby("expiry") if "expiry" in df.columns else [(None, df)]

        for expiry_date, group in expiry_groups:
            group = group.sort_values("strike").copy()

            valid_shorts = group[
                (group["delta"].abs() >= short_delta_range[0]) &
                (group["delta"].abs() <= short_delta_range[1])
            ].copy()

            valid_longs = group[
                (group["delta"].abs() >= long_delta_range[0]) &
                (group["delta"].abs() <= long_delta_range[1])
            ].copy()

            if valid_shorts.empty or valid_longs.empty:
                continue

            for _, short_leg in valid_shorts.iterrows():
                for _, long_leg in valid_longs.iterrows():
                    actual_width = abs(float(short_leg["strike"]) - float(long_leg["strike"]))

                    # Width limits
                    if not (spread_width[0] <= actual_width <= spread_width[1]):
                        continue

                    # Vertical structure rules
                    if is_call:
                        # Call verticals: long strike must be ABOVE short strike
                        if float(long_leg["strike"]) <= float(short_leg["strike"]):
                            continue
                    else:
                        # Put verticals: long strike must be BELOW short strike
                        if float(long_leg["strike"]) >= float(short_leg["strike"]):
                            continue

                    short_bid = float(short_leg.get("bid", 0.0) or 0.0)
                    short_ask = float(short_leg.get("ask", 0.0) or 0.0)
                    long_bid = float(long_leg.get("bid", 0.0) or 0.0)
                    long_ask = float(long_leg.get("ask", 0.0) or 0.0)

                    # Pricing math
                    if is_credit:
                        net_premium = short_bid - long_ask
                        max_loss = actual_width - net_premium
                        max_profit = net_premium
                    else:
                        net_premium = short_ask - long_bid
                        max_loss = net_premium
                        max_profit = actual_width - net_premium

                    if net_premium <= 0 or max_loss <= 0:
                        continue

                    pairs.append({
                        "short_symbol": short_leg["symbol"],
                        "long_symbol": long_leg["symbol"],
                        "short_strike": float(short_leg["strike"]),
                        "long_strike": float(long_leg["strike"]),
                        "dte": int(short_leg["dte"]),
                        "expiry": short_leg.get("expiry"),
                        "short_delta": float(short_leg["delta"]),
                        "long_delta": float(long_leg["delta"]),
                        "net_delta": float(short_leg["delta"]) - float(long_leg["delta"]),
                        "premium": float(net_premium),
                        "max_loss": float(max_loss),
                        "max_profit": float(max_profit),
                        "rr_ratio": float(max_profit / max_loss),
                        "actual_width": float(actual_width),
                        "short_bid": short_bid,
                        "short_ask": short_ask,
                        "long_bid": long_bid,
                        "long_ask": long_ask,
                        "short_strike_dist_pct": abs(float(short_leg["strike"]) - self.spot) / self.spot,
                        "long_strike_dist_pct": abs(float(long_leg["strike"]) - self.spot) / self.spot,
                    })

        out = pd.DataFrame(pairs)
        if out.empty:
            return out

        # Rank best candidates first
        out = out.sort_values(
            ["rr_ratio", "premium", "actual_width"],
            ascending=[False, False, True]
        ).reset_index(drop=True)

        return out
