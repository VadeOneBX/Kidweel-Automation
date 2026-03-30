"""
src/qops/data/sg_context_builder.py
─────────────────────────────────────────────────────────────────────────────
Derive SpotGamma context records from a normalized SpotGamma history DataFrame.

Consumes output of ``qops.data.sg_normalize.normalize()`` only (no parallel ingest).

Computes volatility risk premium (VRP), rolling z-score of VRP, regime labels,
and confidence. No Redis, Alpaca, execution, ORB/UW/confluence, or ranking.

VRP is ``iv_1m - rv_1m`` only when both inputs are present and finite; missing
values are never imputed with zero.

Confidence:
- When VRP cannot be computed (missing ``iv_1m`` and/or ``rv_1m``), confidence
  is derived **only** from ``gamma_ratio`` (distance from 1.0), capped to [0, 1].
- When VRP is available and ``vrp_z`` is available, confidence blends |vrp_z|
  (scaled) with gamma distance.
- When VRP is available but ``vrp_z`` is still warming up (NaN), confidence uses
  |VRP| magnitude (scaled) combined with gamma distance when present.

Regime assignment is a fixed priority rule set (see ``_classify_regime``).
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final

import numpy as np
import pandas as pd

# --- Thresholds (deterministic) ------------------------------------------------
_VRP_Z_SELL: Final[float] = 0.5
_VRP_Z_BUY: Final[float] = -0.5
_VRP_DIRECT_EPS: Final[float] = 0.02
_GAMMA_SQUEEZE: Final[float] = 1.15
_DEFAULT_VRP_Z_WINDOW: Final[int] = 20


class RegimeLabel(StrEnum):
    BUY_PREMIUM = "BUY_PREMIUM"
    SELL_PREMIUM = "SELL_PREMIUM"
    SQUEEZE_UP = "SQUEEZE_UP"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class SpotGammaContext:
    """Typed SpotGamma context for one trade date."""

    ticker: str
    trade_date: date
    vrp: float | None
    vrp_z: float | None
    regime: RegimeLabel
    regime_label: str
    confidence: float
    gamma_ratio: float | None = None
    iv_rank: float | None = None
    iv_1m: float | None = None
    rv_1m: float | None = None
    low_vol_point: float | None = None
    high_vol_point: float | None = None
    implied_move: float | None = None
    notes: str | None = None


def build(
    ticker: str,
    normalized_df: pd.DataFrame,
    *,
    vrp_z_window: int = _DEFAULT_VRP_Z_WINDOW,
) -> list[SpotGammaContext]:
    """
    Build a list of ``SpotGammaContext`` rows, one per calendar row in history.

    Args:
        ticker: Equity or index symbol for this series.
        normalized_df: Output of ``normalize()`` sorted ascending by ``trade_date``.
        vrp_z_window: Rolling window length (rows) for VRP z-score; requires
            at least this many finite VRP points before ``vrp_z`` is defined.

    Returns:
        One context record per input row, same order as ``normalized_df``.

    Raises:
        ValueError: if ``normalized_df`` is empty or missing ``trade_date``.
    """
    if normalized_df.empty:
        raise ValueError("normalized_df must not be empty.")
    if "trade_date" not in normalized_df.columns:
        raise ValueError("normalized_df must contain column 'trade_date'.")

    df = normalized_df.reset_index(drop=True)
    vrp_series = _compute_vrp(df)
    vrp_z_series = _rolling_z(vrp_series, window=vrp_z_window)

    out: list[SpotGammaContext] = []
    for i in range(len(df)):
        row = df.iloc[i]
        td = row["trade_date"]
        if not isinstance(td, date):
            td = pd.Timestamp(td).date()

        vrp = _scalar_float_or_none(vrp_series.iloc[i])
        vrp_z = _scalar_float_or_none(vrp_z_series.iloc[i])

        gr = _col_float(row, "gamma_ratio")
        ivr = _col_float(row, "iv_rank")
        iv1 = _col_float(row, "iv_1m")
        rv1 = _col_float(row, "rv_1m")
        lvp = _col_float(row, "low_vol_point")
        hvp = _col_float(row, "high_vol_point")
        oim = _col_float(row, "options_implied_move")

        regime = _classify_regime(vrp=vrp, vrp_z=vrp_z, gamma_ratio=gr)
        conf = _confidence(vrp=vrp, vrp_z=vrp_z, gamma_ratio=gr)
        notes = _build_notes(vrp=vrp, vrp_z=vrp_z, confidence=conf, gamma_ratio=gr)

        out.append(
            SpotGammaContext(
                ticker=ticker,
                trade_date=td,
                vrp=vrp,
                vrp_z=vrp_z,
                regime=regime,
                regime_label=regime.value,
                confidence=conf,
                gamma_ratio=gr,
                iv_rank=ivr,
                iv_1m=iv1,
                rv_1m=rv1,
                low_vol_point=lvp,
                high_vol_point=hvp,
                implied_move=oim,
                notes=notes,
            )
        )
    return out


def build_latest(
    ticker: str,
    normalized_df: pd.DataFrame,
    *,
    vrp_z_window: int = _DEFAULT_VRP_Z_WINDOW,
) -> SpotGammaContext:
    """
    Build context for the latest row (last row) of ``normalized_df``.

    Expects the same column contract as ``build()``; the last row must exist.

    Raises:
        ValueError: if ``normalized_df`` is empty.
    """
    if normalized_df.empty:
        raise ValueError("normalized_df must not be empty.")
    built = build(ticker, normalized_df, vrp_z_window=vrp_z_window)
    return built[-1]


def _compute_vrp(df: pd.DataFrame) -> pd.Series:
    """VRP = iv_1m - rv_1m; NaN if either leg missing (no zero-fill)."""
    if "iv_1m" not in df.columns or "rv_1m" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    iv = df["iv_1m"].astype("float64")
    rv = df["rv_1m"].astype("float64")
    ok = iv.notna() & rv.notna()
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out.loc[ok] = (iv[ok] - rv[ok]).astype("float64")
    return out


def _rolling_z(series: pd.Series, *, window: int) -> pd.Series:
    """Rolling z-score; NaN until ``window`` finite VRP observations exist."""
    if window < 2:
        raise ValueError("vrp_z_window must be >= 2.")
    s = series.astype("float64")
    m = s.rolling(window=window, min_periods=window).mean()
    sd = s.rolling(window=window, min_periods=window).std(ddof=0)
    z = (s - m) / sd.replace(0.0, np.nan)
    return z


def _classify_regime(
    *,
    vrp: float | None,
    vrp_z: float | None,
    gamma_ratio: float | None,
) -> RegimeLabel:
    """
    Deterministic regime: VRP z-score when available, else direct VRP, else gamma.

    Priority:
    1. Strong VRP richness / cheapness via ``vrp_z`` thresholds.
    2. Else moderate VRP sign via ``vrp`` vs ``_VRP_DIRECT_EPS``.
    3. Else elevated call-side gamma (``gamma_ratio``) → SQUEEZE_UP.
    4. Else NEUTRAL.
    """
    gr = gamma_ratio
    if gr is not None and (not np.isfinite(gr)):
        gr = None

    # 1) z-score path
    if vrp_z is not None and np.isfinite(vrp_z):
        if vrp_z >= _VRP_Z_SELL:
            return RegimeLabel.SELL_PREMIUM
        if vrp_z <= _VRP_Z_BUY:
            return RegimeLabel.BUY_PREMIUM
        if gr is not None and gr >= _GAMMA_SQUEEZE:
            return RegimeLabel.SQUEEZE_UP
        return RegimeLabel.NEUTRAL

    # 2) direct VRP path (warmup or insufficient history for z)
    if vrp is not None and np.isfinite(vrp):
        if vrp > _VRP_DIRECT_EPS:
            return RegimeLabel.SELL_PREMIUM
        if vrp < -_VRP_DIRECT_EPS:
            return RegimeLabel.BUY_PREMIUM
        if gr is not None and gr >= _GAMMA_SQUEEZE:
            return RegimeLabel.SQUEEZE_UP
        return RegimeLabel.NEUTRAL

    # 3) VRP unavailable — only squeeze / neutral from gamma
    if gr is not None and gr >= _GAMMA_SQUEEZE:
        return RegimeLabel.SQUEEZE_UP
    return RegimeLabel.NEUTRAL


def _confidence(
    *,
    vrp: float | None,
    vrp_z: float | None,
    gamma_ratio: float | None,
) -> float:
    """
    Map inputs to [0, 1].

    When ``vrp`` is None (cannot compute VRP from iv_1m/rv_1m), **confidence is
    derived only from ``gamma_ratio``**: ``min(1.0, abs(gamma_ratio - 1.0))``.
    If ``gamma_ratio`` is also missing, returns 0.0.
    """
    gr = gamma_ratio
    if gr is not None and not np.isfinite(gr):
        gr = None

    vrp_missing = vrp is None or (isinstance(vrp, float) and not np.isfinite(vrp))
    if vrp_missing:
        if gr is None:
            return 0.0
        return float(min(1.0, abs(gr - 1.0)))

    # VRP exists
    vz = vrp_z
    vz_ok = vz is not None and isinstance(vz, (float, np.floating)) and np.isfinite(vz)
    gamma_part = float(min(1.0, abs(gr - 1.0))) if gr is not None else 0.0

    if vz_ok:
        z_part = float(min(1.0, abs(float(vz)) / 2.5))
        if gr is not None:
            return float(0.5 * z_part + 0.5 * gamma_part)
        return z_part

    # Warmup: scale |VRP| (typical IV/RV deltas are small decimals)
    v = float(vrp)
    v_part = float(min(1.0, abs(v) * 5.0))
    if gr is not None:
        return float(0.5 * v_part + 0.5 * gamma_part)
    return v_part


def _build_notes(
    *,
    vrp: float | None,
    vrp_z: float | None,
    confidence: float,
    gamma_ratio: float | None,
) -> str | None:
    parts: list[str] = []
    vrp_missing = vrp is None or (isinstance(vrp, float) and not np.isfinite(vrp))
    if vrp_missing:
        parts.append("vrp_not_computed_missing_iv_or_rv")
    if vrp_missing and gamma_ratio is not None and np.isfinite(gamma_ratio):
        parts.append("confidence_from_gamma_ratio_only")
    if vrp_z is None or (isinstance(vrp_z, float) and not np.isfinite(vrp_z)):
        if not vrp_missing:
            parts.append("vrp_z_warmup_or_insufficient_history")
    return "; ".join(parts) if parts else None


def _col_float(row: pd.Series, name: str) -> float | None:
    if name not in row.index:
        return None
    v = row[name]
    if pd.isna(v):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _scalar_float_or_none(x: object) -> float | None:
    if pd.isna(x):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f
