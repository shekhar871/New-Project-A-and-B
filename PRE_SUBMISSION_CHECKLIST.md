# Pre-submission checklist — AgentGuard-Gym

This is the “no surprises” sequence to confirm everything works end-to-end before you submit.

## 1) Environment setup

From `agentguard-gym/`:

```bash
uv sync --extra dev
# For GRPO + optional Unsloth:
uv sync --extra dev --extra grandfinals
```

## 2) Unit tests (math + reward bounds)

```bash
uv run pytest
```

## 3) OpenEnv validation (must pass)

```bash
uv run openenv validate --verbose .
```

## 4) Run the server + UI (real-time episode/training view)

```bash
uv run server
```

Then open:

- `http://127.0.0.1:7860/ui` (dashboard)
- `http://127.0.0.1:7860/docs` (API)

On the dashboard:

- Click **Start trainer** to see episodes and rewards live.
- Click **Run one episode** to inspect step-by-step outcomes.

## 5) Reproducible “before” baseline (no API, no GPU)

```bash
uv run python inference.py
```

Uses keyword heuristics on `data/holdout_attacks.json` and prints a block rate. Judges can run this on a fresh clone.

## 6) Generalization with API (optional)

With `HF_TOKEN` / `API_BASE_URL` / `MODEL_NAME` set:

```bash
uv run python scripts/eval_before_after.py
```

## 7) GRPO training (optional; works without Unsloth)

- Generate corpus: `uv run python scripts/generate_offline_corpus.py`
- Train: `uv run python train_adversarial.py` (uses **Unsloth** if installed, else **transformers + PEFT + 4bit**)
- Metrics land in W&B and in `runs/training_metrics.jsonl` (from `JsonlTrainingMetricsCallback`)
- Plot: `uv run python scripts/plot_training_metrics.py` → `results/training_curve.png` (requires `matplotlib` from `--extra dev`)

## 8) UI “trainer” vs full GRPO

The live UI **trainer** is a **simulated** run (`heuristic` / `random` over the real `AgentGuardEnvironment`) for visibility. **GRPO** is `train_adversarial.py` and produces LoRA weights under `checkpoints/defender/`.
