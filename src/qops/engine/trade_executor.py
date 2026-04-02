"""
src/qops/engine/trade_executor.py
─────────────────────────────────────────────────────────────────────────────
Thin orchestration glue between risk approval and paper execution.

This module does not add decision logic, sizing, or fill modeling. It only:
1) performs minimal pre-submit eligibility checks, then
2) delegates to ``qops.engine.paper_broker.execute_paper_trade``.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from qops.engine.paper_broker import PaperExecutionResult, execute_paper_trade
from qops.engine.risk_guard import RiskApprovalResult
from qops.strategy.spread_builder import BuildOutcome, StructureCandidate


@dataclass(frozen=True)
class TradeExecutionResult:
    """``submitted`` is True only when ``paper_result.executed`` is True."""

    attempted: bool
    submitted: bool
    approved: bool
    ticker: str
    outcome: BuildOutcome | None
    quantity: int
    paper_result: PaperExecutionResult | None
    notes: str


def execute_trade(
    approval: RiskApprovalResult | None,
    candidate: StructureCandidate | None,
    *,
    timestamp: datetime | None = None,
) -> TradeExecutionResult:
    """
    Execute a paper trade attempt for one pre-selected structure candidate.

    Rules:
    - If approval is missing/false, do not submit.
    - If candidate is missing or not ``BULL_CALL_SPREAD``, do not submit.
    - Otherwise call ``execute_paper_trade`` once. ``submitted`` is True only
      when the paper broker reports ``executed`` (full fill).
    """
    attempted = approval is not None and candidate is not None
    precheck = _pre_submit_validation(approval=approval, candidate=candidate)
    if precheck is not None:
        return precheck

    assert approval is not None
    assert candidate is not None
    paper = execute_paper_trade(approval=approval, candidate=candidate, timestamp=timestamp)
    return TradeExecutionResult(
        attempted=attempted,
        submitted=paper.executed,
        approved=approval.approved,
        ticker=candidate.ticker,
        outcome=candidate.outcome,
        quantity=approval.capped_quantity if paper.executed else 0,
        paper_result=paper,
        notes=(
            "submitted_to_paper_broker"
            if paper.executed
            else f"paper_broker_rejected:{paper.rejection_reason or 'unknown'}"
        ),
    )


def _pre_submit_validation(
    *,
    approval: RiskApprovalResult | None,
    candidate: StructureCandidate | None,
) -> TradeExecutionResult | None:
    if approval is None:
        return TradeExecutionResult(
            attempted=False,
            submitted=False,
            approved=False,
            ticker=(candidate.ticker if candidate is not None else ""),
            outcome=(candidate.outcome if candidate is not None else None),
            quantity=0,
            paper_result=None,
            notes="missing_approval",
        )
    if candidate is None:
        return TradeExecutionResult(
            attempted=False,
            submitted=False,
            approved=approval.approved,
            ticker="",
            outcome=None,
            quantity=0,
            paper_result=None,
            notes="missing_candidate",
        )
    if not approval.approved:
        return TradeExecutionResult(
            attempted=True,
            submitted=False,
            approved=False,
            ticker=candidate.ticker,
            outcome=candidate.outcome,
            quantity=0,
            paper_result=None,
            notes="approval_false_not_submitted",
        )
    if candidate.outcome != BuildOutcome.BULL_CALL_SPREAD:
        return TradeExecutionResult(
            attempted=True,
            submitted=False,
            approved=True,
            ticker=candidate.ticker,
            outcome=candidate.outcome,
            quantity=0,
            paper_result=None,
            notes=f"unsupported_candidate_outcome:{candidate.outcome.value}",
        )
    return None

