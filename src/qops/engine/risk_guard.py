"""
src/qops/engine/risk_guard.py
─────────────────────────────────────────────────────────────────────────────
Final engine-layer approval gate before any future paper execution layer.

This module is pure approval logic:
- no Alpaca calls
- no Redis writes
- no order submission
- no broker/execution side effects
- no ORB/UW/confluence logic

It accepts either:
1) a pre-selected structure candidate, or
2) a build result that contains exactly one candidate.

Auto-approval: BULL_CALL_SPREAD and BEAR_PUT_SPREAD only, when EV and risk caps
pass and direction matches the spread (LONG_ONLY / SHORT_ONLY respectively).

Review-only (never auto-approved here): LONG_CALL_PARKED, LONG_PUT_PARKED, SKIP.

Deterministic policy for multiple tradeable candidates:
- reject and require caller pre-selection.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass

from qops.strategy.sg_direction_gate import AllowedDirection, DirectionGateResult
from qops.strategy.spread_builder import BuildOutcome, BuildResult, StructureCandidate

_AUTO_APPROVABLE: frozenset[BuildOutcome] = frozenset(
    {BuildOutcome.BULL_CALL_SPREAD, BuildOutcome.BEAR_PUT_SPREAD}
)


@dataclass(frozen=True)
class RiskApprovalResult:
    approved: bool
    rejection_reasons: tuple[str, ...]
    approved_outcome: BuildOutcome | None
    capped_quantity: int
    adjusted_risk: float
    notes: str | None


def approve(
    *,
    direction: DirectionGateResult,
    selected_candidate: StructureCandidate | None,
    build_result: BuildResult | None,
    max_risk_per_trade: float,
    max_debit_allowed: float,
    max_loss_allowed: float,
    require_ev_pass: bool,
    require_direction_long_only: bool,
) -> RiskApprovalResult:
    """
    Approve or reject one candidate for automated paper execution eligibility.

    Auto-approval: BULL_CALL_SPREAD (LONG_ONLY) or BEAR_PUT_SPREAD (SHORT_ONLY),
    EV-pass, risk caps, ticker match. Parked outcomes and SKIP are rejected.

    If ``require_direction_long_only`` is True, reject unless the direction gate
    is LONG_ONLY (legacy sessions that only permit bullish flow).
    """
    reasons: list[str] = []
    candidate = _extract_candidate(selected_candidate=selected_candidate, build_result=build_result, reasons=reasons)

    if require_direction_long_only and direction.allowed_direction != AllowedDirection.LONG_ONLY:
        reasons.append(f"direction_not_long_only:{direction.allowed_direction.value}")

    if candidate is None:
        return _reject(reasons)

    _check_outcome_eligibility(candidate, reasons)
    _check_direction_compatibility(direction=direction, candidate=candidate, reasons=reasons)
    _check_ev(candidate=candidate, require_ev_pass=require_ev_pass, reasons=reasons)
    _check_risk_caps(
        candidate=candidate,
        max_debit_allowed=max_debit_allowed,
        max_loss_allowed=max_loss_allowed,
        reasons=reasons,
    )

    if max_risk_per_trade <= 0:
        reasons.append("invalid_max_risk_per_trade")

    quantity = _capped_quantity(max_risk_per_trade=max_risk_per_trade, per_unit_risk=candidate.max_loss)
    if quantity < 1:
        reasons.append("risk_cap_allows_zero_quantity")
        return _reject(reasons)

    if reasons:
        return _reject(reasons)

    adjusted_risk = quantity * candidate.max_loss
    return RiskApprovalResult(
        approved=True,
        rejection_reasons=(),
        approved_outcome=candidate.outcome,
        capped_quantity=quantity,
        adjusted_risk=adjusted_risk,
        notes=f"approved_with_quantity={quantity}",
    )


def _extract_candidate(
    *,
    selected_candidate: StructureCandidate | None,
    build_result: BuildResult | None,
    reasons: list[str],
) -> StructureCandidate | None:
    if selected_candidate is not None and build_result is not None:
        reasons.append("provide_selected_candidate_or_build_result_not_both")
        return None
    if selected_candidate is not None:
        return selected_candidate
    if build_result is None:
        reasons.append("missing_candidate_input")
        return None

    if build_result.outcome == BuildOutcome.SKIP:
        reasons.append("build_result_skip")
        return None
    if build_result.outcome == BuildOutcome.LONG_CALL_PARKED:
        reasons.append("build_result_long_call_parked_review_only")
        return None
    if build_result.outcome == BuildOutcome.LONG_PUT_PARKED:
        reasons.append("build_result_long_put_parked_review_only")
        return None
    if build_result.outcome not in _AUTO_APPROVABLE:
        reasons.append(f"unsupported_build_outcome:{build_result.outcome.value}")
        return None
    if not build_result.candidates:
        reasons.append("build_result_has_no_candidates")
        return None
    if len(build_result.candidates) != 1:
        reasons.append("multiple_candidates_require_preselection")
        return None
    return build_result.candidates[0]


def _check_outcome_eligibility(candidate: StructureCandidate, reasons: list[str]) -> None:
    if candidate.outcome == BuildOutcome.LONG_CALL_PARKED:
        reasons.append("long_call_parked_not_auto_approvable")
    elif candidate.outcome == BuildOutcome.LONG_PUT_PARKED:
        reasons.append("long_put_parked_not_auto_approvable")
    elif candidate.outcome == BuildOutcome.SKIP:
        reasons.append("skip_not_auto_approvable")
    elif candidate.outcome not in _AUTO_APPROVABLE:
        reasons.append(f"unsupported_candidate_outcome:{candidate.outcome.value}")


def _check_direction_compatibility(
    *,
    direction: DirectionGateResult,
    candidate: StructureCandidate,
    reasons: list[str],
) -> None:
    ad = direction.allowed_direction
    if ad == AllowedDirection.SKIP:
        reasons.append("direction_gate_skip")
    elif ad == AllowedDirection.LONG_GAMMA_HEDGE:
        reasons.append("direction_gate_long_gamma_hedge_review_only")
    elif ad == AllowedDirection.SHORT_GAMMA_HEDGE:
        reasons.append("direction_gate_short_gamma_hedge_review_only")

    if direction.ticker.strip().upper() != candidate.ticker.strip().upper():
        reasons.append("direction_candidate_ticker_mismatch")

    if candidate.outcome not in _AUTO_APPROVABLE:
        return

    if ad == AllowedDirection.LONG_ONLY and candidate.outcome != BuildOutcome.BULL_CALL_SPREAD:
        reasons.append(f"direction_candidate_incompatible:{ad.value}:{candidate.outcome.value}")
    elif ad == AllowedDirection.SHORT_ONLY and candidate.outcome != BuildOutcome.BEAR_PUT_SPREAD:
        reasons.append(f"direction_candidate_incompatible:{ad.value}:{candidate.outcome.value}")
    elif ad not in (AllowedDirection.LONG_ONLY, AllowedDirection.SHORT_ONLY):
        reasons.append(f"direction_candidate_incompatible:{ad.value}:{candidate.outcome.value}")


def _check_ev(*, candidate: StructureCandidate, require_ev_pass: bool, reasons: list[str]) -> None:
    if not require_ev_pass:
        return
    if candidate.pass_ev_gate is not True:
        reasons.append("candidate_ev_gate_failed")
        return
    if candidate.ev_result is None:
        reasons.append("missing_ev_result")
        return
    if not candidate.ev_result.pass_ev_gate:
        reasons.append("ev_result_gate_failed")


def _check_risk_caps(
    *,
    candidate: StructureCandidate,
    max_debit_allowed: float,
    max_loss_allowed: float,
    reasons: list[str],
) -> None:
    if max_debit_allowed <= 0:
        reasons.append("invalid_max_debit_allowed")
    if max_loss_allowed <= 0:
        reasons.append("invalid_max_loss_allowed")
    if candidate.debit > max_debit_allowed:
        reasons.append("debit_above_cap")
    if candidate.max_loss > max_loss_allowed:
        reasons.append("max_loss_above_cap")
    if candidate.max_loss <= 0:
        reasons.append("candidate_max_loss_non_positive")


def _capped_quantity(*, max_risk_per_trade: float, per_unit_risk: float) -> int:
    if max_risk_per_trade <= 0 or per_unit_risk <= 0:
        return 0
    return int(max_risk_per_trade // per_unit_risk)


def _reject(reasons: list[str]) -> RiskApprovalResult:
    uniq = tuple(dict.fromkeys(reasons))
    return RiskApprovalResult(
        approved=False,
        rejection_reasons=uniq,
        approved_outcome=None,
        capped_quantity=0,
        adjusted_risk=0.0,
        notes="; ".join(uniq) if uniq else "rejected",
    )
