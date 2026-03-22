# Artifact 04 — OG vs INST System Behavior

## Two Design Philosophies

The research evolved into two models:

- **OG** — confluence model
- **INST** — input model

## OG Model

OG relies on internal signal alignment.

Characteristics:

- volume filters
- ORB structure
- stop logic
- risk multipliers
- confluence stacking

Small parameter changes can materially alter results.

### Research Framing

OG behaves like a **configuration-sensitive system**.

Performance depends heavily on:

- parameter selection
- confluence alignment
- stop logic
- risk layering

This increases:

- flexibility
- expressiveness

But also increases:

- overfitting risk
- sensitivity to small tweaks

## INST Model

INST relies more heavily on external inputs.

Characteristics:

- regime detection
- flow signals
- context gating
- environment classification

INST behaves like an **input-sensitive system**.

Performance depends more on:

- signal quality
- environment classification
- external decisioning inputs

and less on internal parameter tuning.

## Core Insight

Two valid system philosophies:

- **OG:** knobs matter
- **INST:** inputs matter

Both are useful.

They answer different questions.

OG is easier to tune inside TradingView.
INST is harder to replicate purely inside TradingView because more of its edge depends on external decision support.
