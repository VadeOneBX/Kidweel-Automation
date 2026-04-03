"""
src/qops/engine/paper_broker.py
─────────────────────────────────────────────────────────────────────────────
Paper-only simulated broker: full fills at candidate debit, no Alpaca, no Redis,
no networking, no async, no persistence. In-memory audit trail only.

Debit-spread simulation only: ``BULL_CALL_SPREAD`` and ``BEAR_PUT_SPREAD`` use
the same dollar model (per-spread debit, max_loss as max_risk, max_profit scaled
by quantity). Rejects unapproved trades, ``SKIP``, parked outcomes, and
outcome/approval mismatches deterministically.

Slippage: none beyond ``StructureCandidate.debit``. No partial fills, no order
states, no routing.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

from qops.engine.risk_guard import RiskApprovalResult
from qops.strategy.spread_builder import BuildOutcome, StructureCandidate

_EXECUTABLE_DEBIT_SPREADS: frozenset[BuildOutcome] = frozenset(
    {BuildOutcome.BULL_CALL_SPREAD, BuildOutcome.BEAR_PUT_SPREAD}
)

# In-memory execution attempt log (both fills and rejections).
_AUDIT: list["PaperExecutionResult"] = []


@dataclass(frozen=True)
class PaperExecutionResult:
    executed: bool
    ticker: str
    structure_type: str
    strikes: tuple[float, float] | None
    expiry: date | None
    quantity: int
    fill_price: float
    total_debit: float
    max_risk: float
    max_profit: float
    notes: str
    timestamp: datetime | None
    rejection_reason: str | None


def execute_paper_trade(
    approval: RiskApprovalResult,
    candidate: StructureCandidate,
    *,
    timestamp: datetime | None = None,
) -> PaperExecutionResult:
    """
    Simulate a single full fill at ``candidate.debit`` per spread for
    ``approval.capped_quantity`` spreads (bull call or bear put debit spread).

    If validation fails, returns ``executed=False`` with a rejection reason.
    Appends every attempt to the in-memory audit trail (success or failure).
    """
    reasons = _validate_approval(approval=approval, candidate=candidate)
    if reasons:
        result = _build_rejection(candidate=candidate, reasons=reasons, timestamp=timestamp)
        _AUDIT.append(result)
        return result

    result = _build_execution_result(
        approval=approval,
        candidate=candidate,
        timestamp=timestamp,
    )
    _AUDIT.append(result)
    return result


def paper_audit_trail() -> tuple[PaperExecutionResult, ...]:
    """
    Copy of all execution attempts recorded this process (in order).

    Includes both successful fills and rejected attempts.
    """
    return tuple(_AUDIT)


def paper_positions_snapshot() -> dict[str, object]:
    """
    Minimal aggregate of **executed** auto-executable debit spreads (in-memory):
    ``BULL_CALL_SPREAD`` and ``BEAR_PUT_SPREAD``.

    Keys: ``by_ticker`` (dict of ticker -> row dict with quantity, total_debit,
    max_risk, max_profit).
    """
    by_ticker: dict[str, dict[str, float | int]] = {}
    for r in _AUDIT:
        if not r.executed:
            continue
        row = by_ticker.setdefault(
            r.ticker,
            {"quantity": 0, "total_debit": 0.0, "max_risk": 0.0, "max_profit": 0.0},
        )
        row["quantity"] = int(row["quantity"]) + r.quantity
        row["total_debit"] = float(row["total_debit"]) + r.total_debit
        row["max_risk"] = float(row["max_risk"]) + r.max_risk
        row["max_profit"] = float(row["max_profit"]) + r.max_profit
    return {"by_ticker": by_ticker}


def reset_paper_audit() -> None:
    """Clear in-memory audit (for tests / deterministic replays)."""
    _AUDIT.clear()


def _validate_approval(
    *,
    approval: RiskApprovalResult,
    candidate: StructureCandidate,
) -> list[str]:
    reasons: list[str] = []
    if not approval.approved:
        reasons.append("approval_not_approved")
    if (
        approval.approved
        and approval.approved_outcome is not None
        and approval.approved_outcome != candidate.outcome
    ):
        reasons.append("approval_outcome_mismatch")
    if candidate.outcome == BuildOutcome.SKIP:
        reasons.append("candidate_outcome_skip")
    elif candidate.outcome == BuildOutcome.LONG_CALL_PARKED:
        reasons.append("candidate_long_call_parked_not_executable")
    elif candidate.outcome == BuildOutcome.LONG_PUT_PARKED:
        reasons.append("candidate_long_put_parked_not_executable")
    elif candidate.outcome not in _EXECUTABLE_DEBIT_SPREADS:
        reasons.append(f"unsupported_structure:{candidate.outcome.value}")
    if approval.capped_quantity < 1:
        reasons.append("invalid_capped_quantity")
    if candidate.debit <= 0:
        reasons.append("invalid_candidate_debit")
    if candidate.max_loss <= 0:
        reasons.append("invalid_candidate_max_loss")
    if candidate.short_strike is None or candidate.width is None:
        reasons.append("debit_spread_missing_short_strike_or_width")
    if candidate.max_profit is None:
        reasons.append("debit_spread_missing_max_profit")

    short = candidate.short_strike
    if candidate.outcome == BuildOutcome.BULL_CALL_SPREAD and short is not None:
        if candidate.long_strike >= short:
            reasons.append("invalid_bull_call_strike_order")
    if candidate.outcome == BuildOutcome.BEAR_PUT_SPREAD and short is not None:
        if candidate.long_strike <= short:
            reasons.append("invalid_bear_put_strike_order")

    return reasons


def _build_execution_result(
    *,
    approval: RiskApprovalResult,
    candidate: StructureCandidate,
    timestamp: datetime | None,
) -> PaperExecutionResult:
    qty: Final[int] = approval.capped_quantity
    fill = float(candidate.debit)
    total_debit = fill * qty
    max_risk = float(candidate.max_loss) * qty
    mp = candidate.max_profit if candidate.max_profit is not None else 0.0
    max_profit = float(mp) * qty
    short = candidate.short_strike
    assert short is not None
    strikes = (float(candidate.long_strike), float(short))
    return PaperExecutionResult(
        executed=True,
        ticker=candidate.ticker.strip().upper(),
        structure_type=candidate.outcome.value,
        strikes=strikes,
        expiry=candidate.expiry,
        quantity=qty,
        fill_price=fill,
        total_debit=total_debit,
        max_risk=max_risk,
        max_profit=max_profit,
        notes="paper_full_fill;no_additional_slippage",
        timestamp=timestamp,
        rejection_reason=None,
    )


def _build_rejection(
    *,
    candidate: StructureCandidate | None,
    reasons: list[str],
    timestamp: datetime | None,
) -> PaperExecutionResult:
    ticker = candidate.ticker.strip().upper() if candidate is not None else ""
    expiry = candidate.expiry if candidate is not None else None
    strikes: tuple[float, float] | None = None
    if candidate is not None and candidate.short_strike is not None:
        strikes = (float(candidate.long_strike), float(candidate.short_strike))
    msg = "; ".join(dict.fromkeys(reasons))
    return PaperExecutionResult(
        executed=False,
        ticker=ticker,
        structure_type=(candidate.outcome.value if candidate is not None else "UNKNOWN"),
        strikes=strikes,
        expiry=expiry,
        quantity=0,
        fill_price=0.0,
        total_debit=0.0,
        max_risk=0.0,
        max_profit=0.0,
        notes="rejected",
        timestamp=timestamp,
        rejection_reason=msg,
    )
