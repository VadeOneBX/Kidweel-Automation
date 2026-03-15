# Constraint Outcomes by Asset

Early experiments compared two execution variants:

INST — deterministic midpoint stop  
OG — wider tolerance with no forced exit time

Results varied significantly across assets.

| Ticker | Variant | Net Profit | Sharpe |
|------|------|------|------|
| SOFI | OG | 8870.56 | 0.456 |
| GME | OG | 6300.13 | 0.474 |
| INTC | INST | 1457.91 | 0.015 |

The conclusion is not that one model dominates.

Different assets express pressure differently.

Trend-driven assets benefited from allowing moves to resolve.

Weaker structures benefited from early invalidation.

This reinforces a broader principle:

Constraints shape system behavior.

The correct constraint depends on the environment.