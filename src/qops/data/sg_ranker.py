"""
src/qops/data/sg_ranker.py
─────────────────────────────────────────────────────────────────────────────
Rank a universe of tickers using latest-row SpotGamma context only.

Input: one ``SpotGammaContext`` per symbol (typically from ``build_latest`` per
ticker). Uses only SpotGamma-derived fields: ``regime_label``, ``vrp_z``,
``confidence``, ``gamma_ratio``.

Output: ``RankedTicker`` rows sorted by ``rank_score`` descending (then ticker).
``structure_bias`` is advisory; execution and structure choice stay downstream.

No Alpaca, no Redis writes, no execution, ORB/UW/confluence, or EV enforcement.
Does not apply Option Alpha reward/risk matrices — optional ``reward_risk_hint``
is a light label only.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from qops.data.sg_context_builder import RegimeLabel, SpotGammaContext

# --- Deterministic thresholds ---------------------------------------------------
_MIN_CONFIDENCE: Final[float] = 0.2
_VRPZ_LONG_CALL: Final[float] = -1.0
_GAMMA_LONG_CALL: Final[float] = 1.25


class StructureBias(StrEnum):
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    LONG_CALL_PARKED = "LONG_CALL_PARKED"
    SKIP = "SKIP"


@dataclass(frozen=True)
class RankedTicker:
    ticker: str
    rank_score: float
    rank_reason: str
    structure_bias: StructureBias
    regime_label: str
    vrp_z: float | None
    confidence: float
    gamma_ratio: float | None
    reward_risk_hint: str | None = None


def rank_latest(contexts: list[SpotGammaContext]) -> list[RankedTicker]:
    """
    Rank tickers from their latest SpotGamma context rows.

    Skip (``rank_score`` 0, ``structure_bias`` SKIP) deterministically when:
    - ``confidence`` is below the minimum threshold
    - ``vrp_z`` is missing (required for scoring)
    - ``regime`` is ``NEUTRAL`` (unsupported for this ranker)
    - ``regime`` is ``SELL_PREMIUM`` (non-cheap premium for buyer-side structures)

    Eligible regimes: ``BUY_PREMIUM`` and ``SQUEEZE_UP`` only.

    Sorting: higher ``rank_score`` first; ties broken by ``ticker`` ascending.
    """
    scored = [_to_ranked(c) for c in contexts]
    return sorted(scored, key=lambda r: (-r.rank_score, r.ticker))


def _to_ranked(ctx: SpotGammaContext) -> RankedTicker:
    regime = ctx.regime
    conf = float(ctx.confidence)
    z = ctx.vrp_z
    gr = ctx.gamma_ratio

    base = RankedTicker(
        ticker=ctx.ticker,
        rank_score=0.0,
        rank_reason="",
        structure_bias=StructureBias.SKIP,
        regime_label=ctx.regime_label,
        vrp_z=z,
        confidence=conf,
        gamma_ratio=gr,
        reward_risk_hint=None,
    )

    if conf < _MIN_CONFIDENCE:
        return _with_skip(base, "skipped_low_confidence")

    if z is None:
        return _with_skip(base, "skipped_missing_vrp_z")

    if regime == RegimeLabel.NEUTRAL:
        return _with_skip(base, "skipped_unsupported_regime")

    if regime == RegimeLabel.SELL_PREMIUM:
        return _with_skip(base, "skipped_non_cheap_premium")

    if regime not in (RegimeLabel.BUY_PREMIUM, RegimeLabel.SQUEEZE_UP):
        return _with_skip(base, "skipped_unsupported_regime")

    score = _compute_score(regime=regime, confidence=conf, vrp_z=z, gamma_ratio=gr)
    bias = _structure_bias(regime=regime, vrp_z=z, gamma_ratio=gr)
    reason = _success_reason(regime=regime)
    hint = _reward_risk_hint(regime=regime, vrp_z=z, gamma_ratio=gr)

    return RankedTicker(
        ticker=ctx.ticker,
        rank_score=score,
        rank_reason=reason,
        structure_bias=bias,
        regime_label=ctx.regime_label,
        vrp_z=z,
        confidence=conf,
        gamma_ratio=gr,
        reward_risk_hint=hint,
    )


def _with_skip(base: RankedTicker, reason: str) -> RankedTicker:
    return RankedTicker(
        ticker=base.ticker,
        rank_score=0.0,
        rank_reason=reason,
        structure_bias=StructureBias.SKIP,
        regime_label=base.regime_label,
        vrp_z=base.vrp_z,
        confidence=base.confidence,
        gamma_ratio=base.gamma_ratio,
        reward_risk_hint=None,
    )


def _compute_score(
    *,
    regime: RegimeLabel,
    confidence: float,
    vrp_z: float,
    gamma_ratio: float | None,
) -> float:
    g = float(gamma_ratio) if gamma_ratio is not None else 1.0
    if regime == RegimeLabel.BUY_PREMIUM:
        # Cheaper IV (more negative z) increases score.
        return float(confidence * (1.0 + max(0.0, -vrp_z)))
    if regime == RegimeLabel.SQUEEZE_UP:
        squeeze_lift = max(0.0, g - 1.0)
        iv_term = 1.0 + max(0.0, -vrp_z) * 0.5
        return float(confidence * (1.0 + 2.0 * squeeze_lift) * iv_term)
    return 0.0


def _structure_bias(
    *,
    regime: RegimeLabel,
    vrp_z: float,
    gamma_ratio: float | None,
) -> StructureBias:
    g = float(gamma_ratio) if gamma_ratio is not None else 1.0
    if regime == RegimeLabel.BUY_PREMIUM:
        if vrp_z <= _VRPZ_LONG_CALL:
            return StructureBias.LONG_CALL_PARKED
        return StructureBias.BULL_CALL_SPREAD
    if regime == RegimeLabel.SQUEEZE_UP:
        if g >= _GAMMA_LONG_CALL:
            return StructureBias.LONG_CALL_PARKED
        return StructureBias.BULL_CALL_SPREAD
    return StructureBias.SKIP


def _success_reason(*, regime: RegimeLabel) -> str:
    if regime == RegimeLabel.BUY_PREMIUM:
        return "ranked_buy_premium"
    if regime == RegimeLabel.SQUEEZE_UP:
        return "ranked_squeeze_up"
    return "ranked"


def _reward_risk_hint(
    *,
    regime: RegimeLabel,
    vrp_z: float,
    gamma_ratio: float | None,
) -> str | None:
    if regime == RegimeLabel.BUY_PREMIUM and vrp_z <= _VRPZ_LONG_CALL:
        return "buyer_favorable_iv_skew"
    if regime == RegimeLabel.SQUEEZE_UP:
        g = gamma_ratio if gamma_ratio is not None else 1.0
        if g >= _GAMMA_LONG_CALL:
            return "upside_gamma_emphasis"
        return "moderate_squeeze_context"
    return None
