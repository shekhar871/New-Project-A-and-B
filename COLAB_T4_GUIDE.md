# Colab (T4) guide — AgentGuard Grand Finals

This is the quickest way to get **real training curves** on a free/cheap **T4** and then re-run the same on Hugging Face Jobs for final submission.

## Why Colab first, HF Jobs for final

- **Colab**: fast iteration (fix config, confirm it trains, get first plots).
- **Hugging Face Jobs (T4)**: reproducible “final run” with clean logs + stable artifact links (what judges expect).

## T4-safe training recipe (data-backed)

These settings are aligned with the Opening Ceremony guidance (T4 is a “good choice”) and with the GRPO practicality constraints in your blueprint:
- **G=4** generations (`num_generations=4`) to avoid OOM
- **batch=1** with **grad accumulation** (default `8`) for stability
- shorter completions (`max_completion_length=200`) to increase steps/hour

## Colab steps (recommended)

1) Start a **GPU** runtime (T4).

2) Clone repo and install:

```bash
git clone https://github.com/shekhar871/Agent_final.git
cd Agent_final/agentguard-gym
pip install -U uv
uv sync --extra dev --extra grandfinals
```

3) Generate offline corpus (600 scenarios):

```bash
uv run python scripts/generate_offline_corpus.py
```

4) Run training (defaults are T4-safe):

```bash
uv run python train_adversarial.py
```

Optional knobs (environment variables):

```bash
export AG_MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
export AG_EPOCHS=1
export AG_GRAD_ACCUM=8
export AG_MAX_COMPLETION=200
export AG_LR=5e-6
```

5) Generate plots (these are the “training evidence” judges want):

```bash
uv run python scripts/plot_training_metrics.py
```

Outputs:
- `plots/reward_curve.png`
- `plots/entropy_curve.png`

## Pre-submission confirmation

From `agentguard-gym/`:

```bash
uv run pytest
uv run openenv validate --verbose .
uv run server
```

Then open the dashboard at `http://127.0.0.1:8000/ui`.

