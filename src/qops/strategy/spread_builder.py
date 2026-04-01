"""
src/qops/strategy/spread_builder.py
─────────────────────────────────────────────────────────────────────────────
Build bullish option structure candidates from pre-supplied option-chain rows.

This module is decision-layer construction only:
- no Alpaca calls
- no Redis writes
- no execution side effects
- no ORB/UW/confluence logic
- no final risk approval

Inputs are expected from upstream layers:
- direction from ``qops.strategy.sg_direction_gate``
- SpotGamma context/ranked context from ``qops.data.*``
- option chain rows already fetched elsewhere

Outputs:
- BULL_CALL_SPREAD
- LONG_CALL_PARKED
- SKIP
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Mapping, Sequence

from qops.data.sg_context_builder import SpotGammaContext
from qops.data.sg_ranker import RankedTicker
from qops.strategy.ev_calculator import EVCalculationResult, EVInputs, calculate
from qops.strategy.sg_direction_gate import AllowedDirection, DirectionGateResult


class BuildOutcome(StrEnum):
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    LONG_CALL_PARKED = "LONG_CALL_PARKED"
    SKIP = "SKIP"


@dataclass(frozen=True)
class ChainCallRow:
    ticker: str
    expiry: date
    dte: int
    strike: float
    bid: float
    ask: float
    underlying_price: float | None = None


@dataclass(frozen=True)
class StructureCandidate:
    outcome: BuildOutcome
    ticker: str
    expiry: date
    long_strike: float
    short_strike: float | None
    width: float | None
    debit: float
    max_profit: float | None
    max_loss: float
    probability_of_profit: float | None
    pass_ev_gate: bool | None
    ev_result: EVCalculationResult | None
    notes: str


@dataclass(frozen=True)
class BuildResult:
    ticker: str
    outcome: BuildOutcome
    reason: str
    candidates: tuple[StructureCandidate, ...]
    notes: str | None


def build(
    *,
    ticker: str,
    direction: DirectionGateResult,
    context: SpotGammaContext | RankedTicker,
    chain_rows: Sequence[Mapping[str, object]],
    min_dte: int = 7,
    max_dte: int = 45,
    probability_of_profit: float | None = None,
    fees: float = 0.0,
    slippage_pct: float = 0.0,
) -> BuildResult:
    """
    Build bullish structure candidates using upstream direction + context.

    Behavior:
    - Direction SKIP -> SKIP (no candidates)
    - Direction LONG_GAMMA_HEDGE -> LONG_CALL_PARKED only
    - Direction LONG_ONLY -> evaluate bull call spread candidates and keep only
      deterministic-valid + EV-pass candidates.
    """
    if direction.allowed_direction == AllowedDirection.SKIP:
        return BuildResult(
            ticker=ticker,
            outcome=BuildOutcome.SKIP,
            reason="direction_gate_skip",
            candidates=(),
            notes=direction.reason,
        )

    calls = _normalize_chain_rows(ticker=ticker, chain_rows=chain_rows)
    if not calls:
        return BuildResult(
            ticker=ticker,
            outcome=BuildOutcome.SKIP,
            reason="no_valid_chain_rows",
            candidates=(),
            notes="malformed_or_empty_chain_rows",
        )

    calls = _filter_by_dte(calls, min_dte=min_dte, max_dte=max_dte)
    if not calls:
        return BuildResult(
            ticker=ticker,
            outcome=BuildOutcome.SKIP,
            reason="no_rows_in_dte_window",
            candidates=(),
            notes=f"min_dte={min_dte}; max_dte={max_dte}",
        )

    if direction.allowed_direction == AllowedDirection.LONG_GAMMA_HEDGE:
        parked = _build_parked_call(ticker=ticker, calls=calls)
        if parked is None:
            return BuildResult(
                ticker=ticker,
                outcome=BuildOutcome.SKIP,
                reason="no_parkable_call",
                candidates=(),
                notes="insufficient_calls_after_filters",
            )
        return BuildResult(
            ticker=ticker,
            outcome=BuildOutcome.LONG_CALL_PARKED,
            reason="direction_gate_long_gamma_hedge",
            candidates=(parked,),
            notes=direction.notes,
        )

    # LONG_ONLY branch
    pop = _resolve_pop(probability_of_profit=probability_of_profit, context=context)
    spread_candidates = _build_spread_candidates(
        ticker=ticker,
        calls=calls,
        probability_of_profit=pop,
        fees=fees,
        slippage_pct=slippage_pct,
    )
    if not spread_candidates:
        return BuildResult(
            ticker=ticker,
            outcome=BuildOutcome.SKIP,
            reason="no_ev_passing_spreads",
            candidates=(),
            notes=f"direction={direction.allowed_direction.value}",
        )

    # Context can shape notes, but must not override direction state.
    regime = _context_regime_label(context)
    note = "squeeze_context_long_only_allowed" if regime == "SQUEEZE_UP" else None
    return BuildResult(
        ticker=ticker,
        outcome=BuildOutcome.BULL_CALL_SPREAD,
        reason="direction_gate_long_only",
        candidates=tuple(spread_candidates),
        notes=note,
    )


def _normalize_chain_rows(*, ticker: str, chain_rows: Sequence[Mapping[str, object]]) -> list[ChainCallRow]:
    out: list[ChainCallRow] = []
    for row in chain_rows:
        opt_type = str(row.get("option_type", row.get("type", "call"))).strip().lower()
        if opt_type not in {"call", "c"}:
            continue
        row_ticker = str(row.get("ticker", ticker)).strip().upper()
        if row_ticker != ticker.upper():
            continue
        expiry = _parse_date(row.get("expiry"))
        strike = _to_float(row.get("strike"))
        bid = _to_float(row.get("bid"))
        ask = _to_float(row.get("ask"))
        if expiry is None or strike is None or bid is None or ask is None:
            continue
        if strike <= 0 or ask <= 0 or bid < 0 or ask < bid:
            continue
        dte_raw = row.get("dte")
        if dte_raw is None:
            continue
        dte = int(dte_raw)
        under = _to_float(row.get("underlying_price"))
        out.append(
            ChainCallRow(
                ticker=row_ticker,
                expiry=expiry,
                dte=dte,
                strike=strike,
                bid=bid,
                ask=ask,
                underlying_price=under,
            )
        )
    return sorted(out, key=lambda r: (r.expiry, r.strike, r.ask, r.bid))


def _filter_by_dte(calls: list[ChainCallRow], *, min_dte: int, max_dte: int) -> list[ChainCallRow]:
    return [r for r in calls if min_dte <= r.dte <= max_dte]


def _build_parked_call(*, ticker: str, calls: list[ChainCallRow]) -> StructureCandidate | None:
    if not calls:
        return None
    under = next((c.underlying_price for c in calls if c.underlying_price is not None), None)
    if under is None:
        mid = calls[len(calls) // 2].strike
    else:
        mid = under
    chosen = min(calls, key=lambda c: (abs(c.strike - mid), c.expiry, c.strike))
    debit = chosen.ask
    return StructureCandidate(
        outcome=BuildOutcome.LONG_CALL_PARKED,
        ticker=ticker,
        expiry=chosen.expiry,
        long_strike=chosen.strike,
        short_strike=None,
        width=None,
        debit=debit,
        max_profit=None,
        max_loss=debit,
        probability_of_profit=None,
        pass_ev_gate=None,
        ev_result=None,
        notes="non_executable_parked_candidate",
    )


def _build_spread_candidates(
    *,
    ticker: str,
    calls: list[ChainCallRow],
    probability_of_profit: float,
    fees: float,
    slippage_pct: float,
) -> list[StructureCandidate]:
    by_expiry: dict[date, list[ChainCallRow]] = {}
    for row in calls:
        by_expiry.setdefault(row.expiry, []).append(row)

    kept: list[StructureCandidate] = []
    for expiry in sorted(by_expiry.keys()):
        rows = sorted(by_expiry[expiry], key=lambda r: r.strike)
        for i in range(len(rows)):
            long_leg = rows[i]
            for j in range(i + 1, len(rows)):
                short_leg = rows[j]
                cand = _spread_candidate(
                    ticker=ticker,
                    long_leg=long_leg,
                    short_leg=short_leg,
                    probability_of_profit=probability_of_profit,
                    fees=fees,
                    slippage_pct=slippage_pct,
                )
                if cand is not None:
                    kept.append(cand)

    kept.sort(
        key=lambda c: (
            -(c.ev_result.expected_value if c.ev_result is not None else -1e9),
            c.expiry,
            c.long_strike,
            c.short_strike if c.short_strike is not None else 0.0,
        )
    )
    return kept


def _spread_candidate(
    *,
    ticker: str,
    long_leg: ChainCallRow,
    short_leg: ChainCallRow,
    probability_of_profit: float,
    fees: float,
    slippage_pct: float,
) -> StructureCandidate | None:
    if long_leg.expiry != short_leg.expiry:
        return None
    if not long_leg.strike < short_leg.strike:
        return None

    debit = long_leg.ask - short_leg.bid
    width = short_leg.strike - long_leg.strike
    if debit <= 0 or width <= 0 or debit >= width:
        return None

    max_profit = width - debit
    max_loss = debit
    ev = calculate(
        EVInputs(
            debit=debit,
            max_profit=max_profit,
            max_loss=max_loss,
            probability_of_profit=probability_of_profit,
            fees=fees,
            slippage_pct=slippage_pct,
            structure_type="bull_call_spread",
        )
    )
    if not ev.pass_ev_gate:
        return None

    return StructureCandidate(
        outcome=BuildOutcome.BULL_CALL_SPREAD,
        ticker=ticker,
        expiry=long_leg.expiry,
        long_strike=long_leg.strike,
        short_strike=short_leg.strike,
        width=width,
        debit=debit,
        max_profit=max_profit,
        max_loss=max_loss,
        probability_of_profit=probability_of_profit,
        pass_ev_gate=ev.pass_ev_gate,
        ev_result=ev,
        notes=ev.notes,
    )


def _resolve_pop(*, probability_of_profit: float | None, context: SpotGammaContext | RankedTicker) -> float:
    if probability_of_profit is not None:
        return _clamp01(probability_of_profit)
    # Deterministic fallback from SpotGamma confidence (conservative range).
    conf = float(context.confidence)
    return _clamp01(0.35 + 0.50 * conf)


def _context_regime_label(context: SpotGammaContext | RankedTicker) -> str:
    return str(context.regime_label)


def _parse_date(v: object) -> date | None:
    if isinstance(v, date):
        return v
    if v is None:
        return None
    try:
        return date.fromisoformat(str(v))
    except ValueError:
        return None


def _to_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)

