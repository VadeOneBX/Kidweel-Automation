from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
import math

import pandas as pd

from qops.data.sg_context_builder import build_latest
from qops.data.sg_normalize import normalize
from qops.data.sg_ranker import rank_latest
from qops.strategy.ev_calculator import EVInputs, calculate
from qops.strategy.sg_direction_gate import evaluate
from qops.strategy.spread_builder import BuildOutcome, build as build_structures


def _build_raw_case(rows: int, *, last_vrp: float, last_gamma_ratio: float) -> pd.DataFrame:
    """Create deterministic raw SpotGamma-like rows as strings for one case."""
    if rows < 20:
        raise ValueError("rows must be >= 20 for rolling z-score behavior")

    start = date(2026, 3, 1)
    trade_dates = [(start + timedelta(days=i)).strftime("%m/%d/%y") for i in range(rows)]

    rv_vals = [0.200 + 0.0005 * i for i in range(rows)]
    # Non-constant VRP history so rolling std is finite.
    vrp_vals = [0.008 + ((i % 5) - 2) * 0.002 for i in range(rows)]
    vrp_vals[-1] = last_vrp
    iv_vals = [rv_vals[i] + vrp_vals[i] for i in range(rows)]
    gamma_vals = [1.05 + 0.01 * ((i % 4) - 1.5) for i in range(rows)]
    gamma_vals[-1] = last_gamma_ratio

    # Keep inputs stringly-typed like CSV export, with apostrophe artifacts.
    data = {
        "Trade Date": trade_dates,
        "1 M IV": [f"'{v:.3f}" for v in iv_vals],
        "1 M RV": [f"'{v:.3f}" for v in rv_vals],
        "Gamma Ratio": [f"'{v:.3f}" for v in gamma_vals],
        "IV Rank": [f"'{40 + i * 0.6:.2f}" for i in range(rows)],
        "Low Vol Point": [f"'{570 + i * 0.4:.2f}" for i in range(rows)],
        "High Vol Point": [f"'{590 + i * 0.4:.2f}" for i in range(rows)],
        "Options Implied Move": [f"'{1.50 + i * 0.01:.2f}" for i in range(rows)],
    }
    return pd.DataFrame(data)


def _build_chain_rows(ticker: str) -> list[dict[str, object]]:
    """Deterministic pre-supplied call chain rows for spread_builder input."""
    return [
        {
            "ticker": ticker,
            "option_type": "call",
            "expiry": "2026-05-15",
            "dte": 21,
            "strike": 580.0,
            "bid": 6.20,
            "ask": 6.40,
            "underlying_price": 582.0,
        },
        {
            "ticker": ticker,
            "option_type": "call",
            "expiry": "2026-05-15",
            "dte": 21,
            "strike": 585.0,
            "bid": 3.95,
            "ask": 4.10,
            "underlying_price": 582.0,
        },
        {
            "ticker": ticker,
            "option_type": "call",
            "expiry": "2026-05-15",
            "dte": 21,
            "strike": 590.0,
            "bid": 2.20,
            "ask": 2.35,
            "underlying_price": 582.0,
        },
    ]


def main() -> None:
    rows = 30
    raw_cases: dict[str, pd.DataFrame] = {
        # Clearly cheap premium (BUY_PREMIUM) and not put-heavy -> LONG_ONLY.
        "CHEAP_LONG_ONLY": _build_raw_case(rows, last_vrp=-0.080, last_gamma_ratio=1.05),
        # Squeeze context (near-neutral VRP z + high gamma ratio) with low confidence -> SKIP.
        "SQUEEZE_SKIP": _build_raw_case(rows, last_vrp=0.008, last_gamma_ratio=1.20),
        # Cheap premium + put-heavy gamma skew -> LONG_GAMMA_HEDGE.
        "CHEAP_HEDGE": _build_raw_case(rows, last_vrp=-0.080, last_gamma_ratio=0.85),
    }

    normalized_cases = {name: normalize(df) for name, df in raw_cases.items()}
    latest_contexts = [
        build_latest(name, normalized_cases[name]) for name in ("CHEAP_LONG_ONLY", "SQUEEZE_SKIP", "CHEAP_HEDGE")
    ]
    ranked = rank_latest(latest_contexts)
    direction_results = [evaluate(item) for item in ranked]
    context_by_ticker = {ctx.ticker: ctx for ctx in latest_contexts}
    direction_by_ticker = {res.ticker: res for res in direction_results}

    build_results = {
        ticker: build_structures(
            ticker=ticker,
            direction=direction_by_ticker[ticker],
            context=context_by_ticker[ticker],
            chain_rows=_build_chain_rows(ticker),
            min_dte=7,
            max_dte=45,
        )
        for ticker in ("CHEAP_LONG_ONLY", "SQUEEZE_SKIP", "CHEAP_HEDGE")
    }

    ev_result = calculate(
        EVInputs(
            debit=2.10,
            max_profit=2.90,
            max_loss=2.10,
            probability_of_profit=0.62,
            fees=0.05,
            slippage_pct=0.03,
            structure_type="bull_call_spread",
        )
    )

    # Fast-fail sanity checks.
    for name, normalized in normalized_cases.items():
        assert len(normalized) >= 20, f"{name}: expected at least 20 rows"
        assert isinstance(normalized.loc[0, "trade_date"], date), f"{name}: trade_date should parse to date"
    assert {ctx.ticker for ctx in latest_contexts} == {"CHEAP_LONG_ONLY", "SQUEEZE_SKIP", "CHEAP_HEDGE"}
    assert len(ranked) == 3, "ranker should return three rows"

    got_directions = {r.allowed_direction.value for r in direction_results}
    assert got_directions == {"LONG_ONLY", "LONG_GAMMA_HEDGE", "SKIP"}, (
        f"expected all three outcomes, got: {sorted(got_directions)}"
    )

    assert math.isfinite(ev_result.expected_value), "EV must be finite"
    assert math.isfinite(ev_result.reward_risk), "reward_risk must be finite"
    assert isinstance(ev_result.pass_ev_gate, bool), "pass_ev_gate must be bool"
    assert build_results["CHEAP_LONG_ONLY"].outcome == BuildOutcome.BULL_CALL_SPREAD
    assert build_results["CHEAP_HEDGE"].outcome == BuildOutcome.LONG_CALL_PARKED
    assert build_results["SQUEEZE_SKIP"].outcome == BuildOutcome.SKIP
    assert len(build_results["CHEAP_LONG_ONLY"].candidates) >= 1
    assert len(build_results["SQUEEZE_SKIP"].candidates) == 0

    print("=== NORMALIZED ===")
    for name in ("CHEAP_LONG_ONLY", "SQUEEZE_SKIP", "CHEAP_HEDGE"):
        print(f"[{name}]")
        print(normalized_cases[name].tail(2).to_string(index=False))
        print()
    print()

    print("=== LATEST CONTEXT ===")
    for ctx in latest_contexts:
        print(asdict(ctx))
    print()

    print("=== RANKED ===")
    for row in ranked:
        print(asdict(row))
    print()

    print("=== DIRECTION GATE ===")
    for row in direction_results:
        print(asdict(row))
    print()

    print("=== EV RESULT ===")
    print(asdict(ev_result))
    print()

    print("=== PACKET 2 FULL PATH ===")
    for ticker in ("CHEAP_LONG_ONLY", "SQUEEZE_SKIP", "CHEAP_HEDGE"):
        print(f"[{ticker}]")
        print("BUILD RESULT")
        print(asdict(build_results[ticker]))
        print("CANDIDATES")
        for candidate in build_results[ticker].candidates:
            print(asdict(candidate))
        if not build_results[ticker].candidates:
            print("[]")
        print()


if __name__ == "__main__":
    main()
