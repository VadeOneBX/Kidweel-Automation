"""
src/qops/engine/ml_router.py
─────────────────────────────────────────────────────────────────────────────
Conservative advisory ML router: optional model score with deterministic
heuristic fallback. Does not override direction gate, EV gate, or risk guard.

No Alpaca, Redis, execution, training, model fitting, or side effects. This
module is easy to bypass or delete: callers can ignore ``pass_ml_gate`` and
rely solely on upstream gates.

Answers: *Given an already-built candidate, does optional model context support
it?* — not whether to trade, how to build, size, or execute.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Final, Literal

from qops.data.sg_context_builder import SpotGammaContext
from qops.data.sg_ranker import RankedTicker
from qops.strategy.spread_builder import BuildOutcome, StructureCandidate

ScoreSource = Literal["model", "heuristic"]

# Advisory only: conservative bar for pass_ml_gate.
_ML_PASS_THRESHOLD: Final[float] = 0.62


@dataclass(frozen=True)
class MLRouteResult:
    candidate: StructureCandidate
    model_score: float
    score_source: ScoreSource
    pass_ml_gate: bool
    notes: str


def route_candidate(
    candidate: StructureCandidate,
    context: SpotGammaContext | RankedTicker | None = None,
) -> MLRouteResult:
    """
    Produce an advisory score and ``pass_ml_gate`` flag.

    Does not mutate ``candidate``. Never submits orders or calls execution.
    """
    model_score: float
    source: ScoreSource
    notes: str

    scored = _try_optional_model_score(candidate=candidate, context=context)
    if scored is not None:
        model_score, source, notes = scored
    else:
        model_score, notes = _heuristic_score(candidate=candidate, context=context)
        source = "heuristic"

    pass_gate = _pass_ml_gate(
        outcome=candidate.outcome,
        score=model_score,
        candidate=candidate,
    )
    return MLRouteResult(
        candidate=candidate,
        model_score=model_score,
        score_source=source,
        pass_ml_gate=pass_gate,
        notes=notes,
    )


def _try_optional_model_score(
    *,
    candidate: StructureCandidate,
    context: SpotGammaContext | RankedTicker | None,
) -> tuple[float, ScoreSource, str] | None:
    if importlib.util.find_spec("qops.ml.candidate_scorer") is not None:
        try:
            mod = importlib.import_module("qops.ml.candidate_scorer")
            fn = getattr(mod, "score_candidate", None)
            if callable(fn):
                raw = fn(candidate, context)
                if raw is not None:
                    s = _clamp01(float(raw))
                    return (s, "model", "score_source=qops.ml.candidate_scorer")
        except (ImportError, AttributeError, TypeError, ValueError):
            pass

    return None


def _heuristic_score(
    *,
    candidate: StructureCandidate,
    context: SpotGammaContext | RankedTicker | None,
) -> tuple[float, str]:
    """Transparent deterministic score in [0, 1]; no EV recomputation."""
    if candidate.outcome == BuildOutcome.SKIP:
        return 0.0, "heuristic_skip_outcome"

    if candidate.outcome == BuildOutcome.LONG_CALL_PARKED:
        return 0.35, "heuristic_parked_review_only"

    if candidate.outcome != BuildOutcome.BULL_CALL_SPREAD:
        return 0.0, f"heuristic_unsupported_outcome:{candidate.outcome.value}"

    score = 0.52
    if candidate.pass_ev_gate is True:
        score += 0.12
        notes_ev = "ev_gate_true"
    elif candidate.pass_ev_gate is False:
        score -= 0.08
        notes_ev = "ev_gate_false"
    else:
        notes_ev = "ev_gate_unknown"

    w = candidate.width
    if w is not None and w > 0 and candidate.debit > 0:
        debit_to_width = candidate.debit / w
        score -= min(0.12, 0.06 * debit_to_width)

    score += _context_bonus(context)

    score = _clamp01(score)
    return score, f"heuristic_bull_call_spread;{notes_ev}"


def _context_bonus(context: SpotGammaContext | RankedTicker | None) -> float:
    if context is None:
        return 0.0
    if float(context.confidence) >= 0.72:
        return 0.02
    return 0.0


def _pass_ml_gate(
    *,
    outcome: BuildOutcome,
    score: float,
    candidate: StructureCandidate,
) -> bool:
    if outcome == BuildOutcome.SKIP:
        return False
    if outcome == BuildOutcome.LONG_CALL_PARKED:
        return False
    if outcome != BuildOutcome.BULL_CALL_SPREAD:
        return False
    if candidate.pass_ev_gate is False:
        return False
    return score >= _ML_PASS_THRESHOLD


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
