# Artifact 01 — Exit Constraint Removal

## Original Assumption

Trades were forced to exit before the mid-day session.

Rationale:

- reduce noise
- avoid liquidity decay
- standardize duration

## Original Logic

```python
if time >= session_exit:
    exit_trade()
```

## Modified Logic

```python
# forced session exit removed
# trades allowed to resolve naturally
```

## Observation

The forced exit constraint stabilized trade duration but truncated continuation.

Removing the constraint changed:

- trade duration
- tail winners
- drawdown profile
- expectancy distribution

## Result

This was not treated as optimization.

It was treated as a test of whether the constraint distorted outcomes.

## Interpretation

The constraint changed system behavior more than expected.

The lesson is broader than trading:

**constraints can distort results as much as signals do.**
