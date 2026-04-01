"""
src/qops/strategy/ev_calculator.py
─────────────────────────────────────────────────────────────────────────────
Option Alpha–style expected value and reward/risk math for **bullish** option
structures. Supports only **debit-defined bullish structures**. Pure functions
only: no Alpaca, Redis, execution, ORB/UW/confluence,
ranking, structure selection, or ML.

Answers: **Does the structure math justify the trade?** — not *whether* to trade,
which ticker, how to size, or how to build the chain.

**EV (binary outcome model)**::

    EV = P(win) * adjusted_max_profit - (1 - P(win)) * adjusted_max_loss

where ``P(win)`` is ``probability_of_profit`` (PoP) supplied by the caller.

**Reward/risk**::

    reward_risk = adjusted_max_profit / adjusted_max_loss   (if max_loss > 0)

**Adjusted values** (conservative, transparent):

* ``adjusted_debit = debit * (1 + slippage_pct) + fees``
* ``adjusted_max_loss = adjusted_debit`` for risk-defined debit structures
  (``bull_call_spread``, ``long_call_hedge``); i.e. max loss equals net debit.
  ``max_loss`` input is accepted for audit/compatibility but not used as the
  adjusted loss source for supported structures.
* ``adjusted_max_profit = max(0, max_profit * (1 - slippage_pct) - fees * 0.5)``
  — slippage haircuts profit; half of flat ``fees`` applied to profit leg (deterministic split).

**Option Alpha reward/risk matrix** (band → minimum required RR) is implemented
as an explicit step table in ``_MIN_RR_BY_POP_BAND``; change only that table to
retune policy.

**profit_to_debit_multiple**:
``adjusted_max_profit / adjusted_debit`` when ``adjusted_debit > 0`` (not a
strike-distance breakeven; caller supplies dollar P/L fields).
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

# Conservative EV gate: require strictly positive EV after float noise.
_EV_EPS: Final[float] = 1e-9
_RR_EPS: Final[float] = 1e-9


class StructureType(StrEnum):
    """Supported structures for EV evaluation."""

    BULL_CALL_SPREAD = "bull_call_spread"
    LONG_CALL_HEDGE = "long_call_hedge"


@dataclass(frozen=True)
class EVInputs:
    """Dollar amounts in a single consistent unit (e.g. per-spread or total)."""

    debit: float
    max_profit: float
    max_loss: float
    probability_of_profit: float
    fees: float = 0.0
    slippage_pct: float = 0.0
    structure_type: str = StructureType.BULL_CALL_SPREAD


@dataclass(frozen=True)
class EVCalculationResult:
    expected_value: float
    reward_risk: float
    profit_to_debit_multiple: float
    adjusted_debit: float
    adjusted_max_profit: float
    adjusted_max_loss: float
    pass_ev_gate: bool
    notes: str
    min_required_reward_risk: float
    meets_reward_risk_rule: bool

    @property
    def breakeven_multiple(self) -> float:
        """Backward-compatible alias for ``profit_to_debit_multiple``."""
        return self.profit_to_debit_multiple


def calculate(inputs: EVInputs) -> EVCalculationResult:
    """
    Compute EV, reward/risk, OA matrix check, and a conservative pass flag.

    Unsupported ``structure_type`` returns zeros/false with a deterministic note
    (no exceptions).
    """
    st = _parse_structure_type(inputs.structure_type)
    if st is None:
        return EVCalculationResult(
            expected_value=0.0,
            reward_risk=0.0,
            profit_to_debit_multiple=0.0,
            adjusted_debit=0.0,
            adjusted_max_profit=0.0,
            adjusted_max_loss=0.0,
            pass_ev_gate=False,
            notes="unsupported_structure_type; use bull_call_spread or long_call_hedge",
            min_required_reward_risk=float("inf"),
            meets_reward_risk_rule=False,
        )

    pop = _clamp_pop(inputs.probability_of_profit)
    adj_debit, adj_profit, adj_loss = _adjusted_pl(inputs)

    if adj_debit <= 0 or adj_loss <= 0:
        return EVCalculationResult(
            expected_value=0.0,
            reward_risk=0.0,
            profit_to_debit_multiple=0.0,
            adjusted_debit=adj_debit,
            adjusted_max_profit=adj_profit,
            adjusted_max_loss=adj_loss,
            pass_ev_gate=False,
            notes="invalid_inputs: adjusted_debit_or_max_loss_non_positive",
            min_required_reward_risk=min_required_reward_risk(pop),
            meets_reward_risk_rule=False,
        )

    ev = _expected_value_binary(pop, adj_profit, adj_loss)
    rr = _reward_risk(adj_profit, adj_loss)
    be_mult = _profit_to_debit_multiple(adj_profit, adj_debit)
    min_rr = min_required_reward_risk(pop)
    meets_rr = rr + _RR_EPS >= min_rr
    pass_ev = ev > _EV_EPS and meets_rr

    notes = _build_notes(
        ev=ev,
        rr=rr,
        min_rr=min_rr,
        meets_rr=meets_rr,
        st=st,
    )
    if abs(float(inputs.max_loss) - float(inputs.debit)) > 1e-6 * max(1.0, abs(float(inputs.debit))):
        notes += "; input_max_loss_debit_mismatch"

    return EVCalculationResult(
        expected_value=ev,
        reward_risk=rr,
        profit_to_debit_multiple=be_mult,
        adjusted_debit=adj_debit,
        adjusted_max_profit=adj_profit,
        adjusted_max_loss=adj_loss,
        pass_ev_gate=pass_ev,
        notes=notes,
        min_required_reward_risk=min_rr,
        meets_reward_risk_rule=meets_rr,
    )


# --- Option Alpha matrix: PoP band -> minimum required reward/risk ---------------
# reward_risk = adjusted_max_profit / adjusted_max_loss
# Lower PoP -> require higher RR. Bands are [low, high) on PoP except last.
_POP_BANDS: Final[tuple[tuple[float, float, float], ...]] = (
    # (pop_low, pop_high_exclusive, min_rr)
    (0.00, 0.25, 4.00),
    (0.25, 0.40, 3.00),
    (0.40, 0.55, 2.00),
    (0.55, 0.70, 1.50),
    (0.70, 1.01, 1.00),
)


def min_required_reward_risk(probability_of_profit: float) -> float:
    """Deterministic matrix lookup: minimum RR required for the PoP band."""
    p = _clamp_pop(probability_of_profit)
    for low, high, min_rr in _POP_BANDS:
        if low <= p < high:
            return min_rr
    return 4.00


def adjusted_values(
    debit: float,
    max_profit: float,
    max_loss: float,
    *,
    fees: float = 0.0,
    slippage_pct: float = 0.0,
    structure_type: str = StructureType.BULL_CALL_SPREAD,
) -> tuple[float, float, float]:
    """
    Return ``(adjusted_debit, adjusted_max_profit, adjusted_max_loss)``.

    For supported debit-defined structures, ``adjusted_max_loss`` is set to
    ``adjusted_debit`` (risk-defined max loss equals net debit).
    """
    inp = EVInputs(
        debit=debit,
        max_profit=max_profit,
        max_loss=max_loss,
        probability_of_profit=0.5,
        fees=fees,
        slippage_pct=slippage_pct,
        structure_type=structure_type,
    )
    return _adjusted_pl(inp)


def _parse_structure_type(raw: str) -> StructureType | None:
    s = raw.strip().lower().replace("-", "_")
    try:
        return StructureType(s)
    except ValueError:
        return None


def _clamp_pop(p: float) -> float:
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return float(p)


def _adjusted_pl(inp: EVInputs) -> tuple[float, float, float]:
    """Slippage widens debit; fees add to debit; profit is haircut."""
    slip = max(0.0, float(inp.slippage_pct))
    fees = max(0.0, float(inp.fees))
    debit = max(0.0, float(inp.debit))
    mxp = max(0.0, float(inp.max_profit))

    adjusted_debit = debit * (1.0 + slip) + fees
    # Conservative: haircut max profit by slippage; half of flat fees from profit side.
    adjusted_max_profit = max(0.0, mxp * (1.0 - slip) - fees * 0.5)
    # Risk-defined bullish debit structures: max loss equals net debit after costs.
    adjusted_max_loss = adjusted_debit
    return (adjusted_debit, adjusted_max_profit, adjusted_max_loss)


def _expected_value_binary(pop: float, max_profit: float, max_loss: float) -> float:
    return pop * max_profit - (1.0 - pop) * max_loss


def _reward_risk(max_profit: float, max_loss: float) -> float:
    if max_loss <= 0:
        return 0.0
    return max_profit / max_loss


def _profit_to_debit_multiple(max_profit: float, debit: float) -> float:
    if debit <= 0:
        return 0.0
    return max_profit / debit


def _build_notes(
    *,
    ev: float,
    rr: float,
    min_rr: float,
    meets_rr: bool,
    st: StructureType,
) -> str:
    parts = [
        f"structure={st.value}",
        f"ev={ev:.6f}",
        f"reward_risk={rr:.6f}",
        f"min_required_rr={min_rr:.6f}",
        f"meets_reward_risk_rule={meets_rr}",
    ]
    if ev <= _EV_EPS:
        parts.append("ev_not_positive")
    if not meets_rr:
        parts.append("reward_risk_below_matrix")
    return "; ".join(parts)
