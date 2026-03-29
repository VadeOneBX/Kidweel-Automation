# Decision Systems Research Lab

![Research](https://img.shields.io/badge/type-research-blue)
![Status](https://img.shields.io/badge/status-active-lightgrey)
![Focus](https://img.shields.io/badge/focus-decision--systems-black)
![Stack](https://img.shields.io/badge/stack-python%20%7C%20redis%20%7C%20alpaca-green)
![Scope](https://img.shields.io/badge/scope-regime--conditioned-orange)

This repository documents a research environment for studying how signal quality changes across operating conditions.

The lab began with intraday market structure research and evolved into a broader framework for:

- environment classification
- signal governance
- constraint testing
- automation readiness

The goal is not building a trading bot.

The goal is building **repeatable decision systems under uncertainty**.

## Research Questions

This lab focuses on three core questions:

1. Does environment classification change signal quality?
2. Do constraints distort decision outcomes?
3. When should automation replace discretion?

## Featured Artifacts

- **Artifact 01 — Exit Constraint Removal**  
  A small rule change materially altered the equity curve.

- **Artifact 02 — Regime Divergence**  
  The same signal produces different outcomes depending on environment.

- **Artifact 03 — Signal Governance**  
  A layered signal stack with escalation and kill-switch logic.

- **Artifact 04 — Manual to Automated Path**  
  How repeatable workflow checkpoints became automation architecture candidates.

- **Artifact 05 — OG vs INST**  
  A comparison between a configuration-sensitive model and an environment-sensitive model.

## Methodology

The framework separates:

**Environment → Signal → Execution**

Rather than optimizing signals directly, behavior is analyzed across regimes and rule sets.

### OG Model

- confluence-heavy
- TradingView-native
- parameter-sensitive
- expressive, but vulnerable to overfitting

### INST Model

- external-input-dependent
- regime and flow gated
- less parameter-sensitive
- harder to replicate purely inside TradingView

## Why This Matters

The same principle applies beyond markets:

- paid media environments change
- auction conditions change
- attribution confidence changes
- user behavior changes

Signals must be evaluated within environment.

## Repository Structure

```text
docs/
  artifacts/
  audiences/
research/
diagrams/
scripts/
src/
```

## Audience

This project is relevant to:

- fintech / trading research
- growth & paid media experimentation
- product analytics
- MBA admissions
- decision science teams

## Architecture

| Layer | Stack |
| --- | --- |
| Research | Python / pandas |
| Backtesting | Custom ORB engine |
| State | Redis |
| Execution | Alpaca API |
| Context | MCP |
| Dev | Cursor |
| Docs | Markdown |

## Navigation

- [Lab overview](docs/overview.md)
- [Research timeline](docs/research_timeline.md) — narrative spine
- [Methodology](docs/methodology.md)
- [Artifacts (evidence shelf)](docs/artifacts/)
- [Audience lenses](docs/audiences/)
