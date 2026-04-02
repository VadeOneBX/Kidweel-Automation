"""
src/qops/engine/run_session.py
─────────────────────────────────────────────────────────────────────────────
Top-level deterministic session orchestrator: wires normalize → context → rank
→ direction → spread build (includes EV) → ML advisory → risk approval →
optional paper execution.

No Alpaca fetching, no Redis, no ORB/UW/confluence. No new business rules;
delegates to existing modules only.

``execute_paper=False`` (default): never calls ``trade_executor`` / paper broker;
``execution_results`` is empty and session ``notes`` record that.

Candidate policy: if a ticker produces more than one spread candidate, ML / risk /
execution are skipped for that ticker (caller must pre-select one candidate).
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

import pandas as pd

from qops.data.sg_context_builder import build_latest
from qops.data.sg_normalize import normalize
from qops.data.sg_ranker import RankedTicker, rank_latest
from qops.engine.ml_router import MLRouteResult, route_candidate
from qops.engine.risk_guard import RiskApprovalResult, approve
from qops.engine.trade_executor import TradeExecutionResult, execute_trade
from qops.strategy.sg_direction_gate import DirectionGateResult, evaluate
from qops.strategy.spread_builder import BuildOutcome, BuildResult, StructureCandidate, build


@dataclass(frozen=True)
class RiskRunParams:
    """Explicit risk guard inputs (no hidden semantics)."""

    max_risk_per_trade: float
    max_debit_allowed: float
    max_loss_allowed: float
    require_ev_pass: bool
    require_direction_long_only: bool


@dataclass(frozen=True)
class SpreadRunParams:
    """Spread builder inputs passed through unchanged."""

    min_dte: int
    max_dte: int
    fees: float
    slippage_pct: float
    probability_of_profit: float | None


@dataclass(frozen=True)
class SessionRunResult:
    started_at: datetime
    tickers_considered: tuple[str, ...]
    ranked_tickers: tuple[RankedTicker, ...]
    direction_results: dict[str, DirectionGateResult]
    build_results: dict[str, BuildResult]
    ml_results: dict[str, MLRouteResult | None]
    approval_results: dict[str, RiskApprovalResult | None]
    execution_results: dict[str, TradeExecutionResult | None]
    notes: str


def run_session(
    *,
    spotgamma_by_ticker: Mapping[str, pd.DataFrame],
    chains_by_ticker: Mapping[str, Sequence[Mapping[str, object]]],
    execute_paper: bool = False,
    risk_params: RiskRunParams | None = None,
    spread_params: SpreadRunParams | None = None,
    started_at: datetime | None = None,
) -> SessionRunResult:
    """
    Run one coordinated session over a small supplied universe.

    * ``spotgamma_by_ticker``: raw SpotGamma export or already-normalized frame
      per ticker (see ``_ensure_normalized``).
    * ``chains_by_ticker``: pre-supplied option chain rows per ticker; missing
      key → empty chain.
    """
    ts = started_at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        raise ValueError("started_at must be timezone-aware")
    rp = risk_params or RiskRunParams(
        max_risk_per_trade=5_000.0,
        max_debit_allowed=5_000.0,
        max_loss_allowed=5_000.0,
        require_ev_pass=True,
        require_direction_long_only=True,
    )
    sp = spread_params or SpreadRunParams(
        min_dte=7,
        max_dte=45,
        fees=0.0,
        slippage_pct=0.0,
        probability_of_profit=None,
    )

    tickers = tuple(sorted(spotgamma_by_ticker.keys()))
    normalized_map = {t: _ensure_normalized(spotgamma_by_ticker[t]) for t in tickers}
    contexts = {t: build_latest(t, normalized_map[t]) for t in tickers}
    ranked = tuple(rank_latest([contexts[t] for t in tickers]))
    ranked_by_ticker = {r.ticker: r for r in ranked}

    direction_results: dict[str, DirectionGateResult] = {}
    build_results: dict[str, BuildResult] = {}
    ml_results: dict[str, MLRouteResult | None] = {}
    approval_results: dict[str, RiskApprovalResult | None] = {}
    execution_results: dict[str, TradeExecutionResult | None] = {}

    note_parts: list[str] = []
    if not execute_paper:
        note_parts.append("execute_paper_disabled")

    for t in tickers:
        direction_results[t] = evaluate(ranked_by_ticker[t])
        chain_rows = chains_by_ticker.get(t) or ()
        build_results[t] = build(
            ticker=t,
            direction=direction_results[t],
            context=contexts[t],
            chain_rows=chain_rows,
            min_dte=sp.min_dte,
            max_dte=sp.max_dte,
            probability_of_profit=sp.probability_of_profit,
            fees=sp.fees,
            slippage_pct=sp.slippage_pct,
        )

        br = build_results[t]
        sel = _select_single_candidate_or_none(br)
        if sel is None:
            if br.outcome == BuildOutcome.BULL_CALL_SPREAD and len(br.candidates) > 1:
                note_parts.append(f"{t}:multiple_spread_candidates_skipped_ml_risk_exec")
            elif br.outcome == BuildOutcome.LONG_CALL_PARKED:
                note_parts.append(f"{t}:parked_call_review_only")
            elif br.outcome == BuildOutcome.SKIP:
                note_parts.append(f"{t}:build_skip")
            ml_results[t] = None
            approval_results[t] = None
            continue

        ml_results[t] = route_candidate(sel, context=contexts[t])
        approval_results[t] = approve(
            direction=direction_results[t],
            selected_candidate=sel,
            build_result=None,
            max_risk_per_trade=rp.max_risk_per_trade,
            max_debit_allowed=rp.max_debit_allowed,
            max_loss_allowed=rp.max_loss_allowed,
            require_ev_pass=rp.require_ev_pass,
            require_direction_long_only=rp.require_direction_long_only,
        )

        if not execute_paper:
            continue

        ex = execute_trade(
            approval=approval_results[t],
            candidate=sel,
            timestamp=ts,
        )
        execution_results[t] = ex

    notes = "; ".join(dict.fromkeys(note_parts)) if note_parts else "ok"
    return SessionRunResult(
        started_at=ts,
        tickers_considered=tickers,
        ranked_tickers=ranked,
        direction_results=direction_results,
        build_results=build_results,
        ml_results=ml_results,
        approval_results=approval_results,
        execution_results=execution_results if execute_paper else {},
        notes=notes,
    )


def _ensure_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Raw export has ``Trade Date``; normalized has ``trade_date``."""
    if df.empty:
        raise ValueError("SpotGamma DataFrame must not be empty.")
    cols = {c.strip() for c in df.columns}
    if "trade_date" in cols:
        return df.copy()
    if "Trade Date" in cols:
        return normalize(df)
    raise ValueError("Expected raw 'Trade Date' or normalized 'trade_date' column.")


def _select_single_candidate_or_none(br: BuildResult) -> StructureCandidate | None:
    if len(br.candidates) != 1:
        return None
    return br.candidates[0]
