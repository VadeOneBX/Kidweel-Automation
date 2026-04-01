"""
src/qops/strategy/sg_direction_gate.py
─────────────────────────────────────────────────────────────────────────────
SpotGamma-driven direction gate: map latest context into an allowed stance
(LONG_ONLY, LONG_GAMMA_HEDGE, or SKIP).

Uses only SpotGamma-derived fields (``regime_label``, ``confidence``, ``vrp_z``,
``gamma_ratio``) from ``SpotGammaContext`` or downstream ``RankedTicker``. No
sizing, structure selection, Alpaca, Redis writes, execution, ORB/UW/confluence,
or EV enforcement — classification only.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from qops.data.sg_context_builder import RegimeLabel, SpotGammaContext
from qops.data.sg_ranker import RankedTicker

# --- Deterministic thresholds ---------------------------------------------------
_MIN_CONF: Final[float] = 0.2
# Call/put gamma ratio below this implies put-side dominance → hedge path.
_GAMMA_HEDGE_MAX: Final[float] = 0.95
# Unstable-vol hedge: requires finite vrp_z >= this (no hedge from vrp_z alone if missing).
_VRPZ_UNSTABLE_MIN: Final[float] = 0.75


class AllowedDirection(StrEnum):
    LONG_ONLY = "LONG_ONLY"
    LONG_GAMMA_HEDGE = "LONG_GAMMA_HEDGE"
    SKIP = "SKIP"


@dataclass(frozen=True)
class DirectionGateResult:
    ticker: str
    allowed_direction: AllowedDirection
    reason: str
    confidence: float
    regime_label: str
    notes: str | None


def classify_direction(source: SpotGammaContext | RankedTicker) -> DirectionGateResult:
    """
    Classify allowed direction from the latest SpotGamma context or a
    ``RankedTicker`` carrying the same fields.

    **SKIP** — ``confidence`` < threshold, unknown ``regime_label``, or
    ``NEUTRAL`` / ``SELL_PREMIUM``; also when ``vrp_z`` is missing for
    **LONG_ONLY** (nontrivial long path requires a VRP z-score).

    **LONG_GAMMA_HEDGE** — ``BUY_PREMIUM`` or ``SQUEEZE_UP``, confidence OK, and
    either put-heavy ``gamma_ratio`` **or** (required) ``vrp_z`` present with
    ``vrp_z >=`` unstable threshold. Unstable vol never triggers hedge without
    finite ``vrp_z``.

    **LONG_ONLY** — same supportive regimes, confidence OK, not in the hedge
    branch, and ``vrp_z`` present.
    """
    ticker, regime_label, confidence, vrp_z, gamma_ratio = _fields(source)
    regime = _parse_regime(regime_label)

    if confidence < _MIN_CONF:
        return DirectionGateResult(
            ticker=ticker,
            allowed_direction=AllowedDirection.SKIP,
            reason="low_confidence",
            confidence=confidence,
            regime_label=regime_label,
            notes=_note_vrp_z(vrp_z),
        )

    if regime is None:
        return DirectionGateResult(
            ticker=ticker,
            allowed_direction=AllowedDirection.SKIP,
            reason="unknown_regime",
            confidence=confidence,
            regime_label=regime_label,
            notes=_note_vrp_z(vrp_z),
        )

    if regime in (RegimeLabel.NEUTRAL, RegimeLabel.SELL_PREMIUM):
        return DirectionGateResult(
            ticker=ticker,
            allowed_direction=AllowedDirection.SKIP,
            reason="unsupported_regime",
            confidence=confidence,
            regime_label=regime_label,
            notes=_note_vrp_z(vrp_z),
        )

    if regime not in (RegimeLabel.BUY_PREMIUM, RegimeLabel.SQUEEZE_UP):
        return DirectionGateResult(
            ticker=ticker,
            allowed_direction=AllowedDirection.SKIP,
            reason="unsupported_regime",
            confidence=confidence,
            regime_label=regime_label,
            notes=_note_vrp_z(vrp_z),
        )

    put_heavy = gamma_ratio is not None and gamma_ratio < _GAMMA_HEDGE_MAX
    unstable_vol = vrp_z is not None and float(vrp_z) >= _VRPZ_UNSTABLE_MIN
    is_hedge = put_heavy or unstable_vol

    if is_hedge:
        note = _hedge_notes(gamma_ratio, vrp_z, put_heavy, unstable_vol)
        return DirectionGateResult(
            ticker=ticker,
            allowed_direction=AllowedDirection.LONG_GAMMA_HEDGE,
            reason="negative_gamma_or_unstable_vol",
            confidence=confidence,
            regime_label=regime_label,
            notes=note,
        )

    if vrp_z is None:
        return DirectionGateResult(
            ticker=ticker,
            allowed_direction=AllowedDirection.SKIP,
            reason="vrp_z_missing",
            confidence=confidence,
            regime_label=regime_label,
            notes="vrp_z_required_for_long_only",
        )

    return DirectionGateResult(
        ticker=ticker,
        allowed_direction=AllowedDirection.LONG_ONLY,
        reason="bullish_regime_sufficient_confidence",
        confidence=confidence,
        regime_label=regime_label,
        notes=None,
    )


def _fields(source: SpotGammaContext | RankedTicker) -> tuple[str, str, float, float | None, float | None]:
    if isinstance(source, SpotGammaContext):
        return (
            source.ticker,
            source.regime_label,
            float(source.confidence),
            source.vrp_z,
            source.gamma_ratio,
        )
    if isinstance(source, RankedTicker):
        return (
            source.ticker,
            source.regime_label,
            float(source.confidence),
            source.vrp_z,
            source.gamma_ratio,
        )
    raise TypeError(f"Expected SpotGammaContext or RankedTicker, got {type(source)!r}")


def _parse_regime(regime_label: str) -> RegimeLabel | None:
    try:
        return RegimeLabel(regime_label.strip())
    except ValueError:
        return None


def _note_vrp_z(vrp_z: float | None) -> str | None:
    if vrp_z is None:
        return "vrp_z_missing"
    return None


def _hedge_notes(
    gamma_ratio: float | None,
    vrp_z: float | None,
    put_heavy: bool,
    unstable_vol: bool,
) -> str:
    parts: list[str] = []
    if put_heavy:
        parts.append(f"put_heavy_gamma_ratio={gamma_ratio}")
    if unstable_vol:
        parts.append(f"vrp_z_unstable>={_VRPZ_UNSTABLE_MIN} (vrp_z={vrp_z})")
    return "; ".join(parts)

