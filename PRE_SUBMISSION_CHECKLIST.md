# Pre-submission checklist — AgentGuard-Gym

This is the “no surprises” sequence to confirm everything works end-to-end before you submit.

## 1) Environment setup

From `agentguard-gym/`:

```bash
uv sync --extra dev
```

## 2) Unit tests (math + reward bounds)

```bash
uv run pytest
```

## 3) OpenEnv validation (must pass)

```bash
uv run openenv validate --verbose .
```

## 4) Run the server + UI (real-time view)

```bash
uv run server
```

Then open:

- `http://127.0.0.1:8000/ui` (dashboard)
- `http://127.0.0.1:8000/docs` (API)

On the dashboard:

- Click **Start trainer** to see episodes and rewards live.
- Click **Run one episode** to inspect step-by-step outcomes.
- For Grand Finals adversarial episodes, call `POST /adversarial/episode` from `/docs` (or wire it from the UI).

## 5) Baseline inference (LLM) — optional

Requires a token:

```bash
export HF_TOKEN=...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
uv run python inference.py
```

## 6) Grand Finals: run training + produce plots (required by judges)

The Opening Ceremony minimum submission requirements include:
- a working training script (TRL/Unsloth)
- evidence of a real training run (reward/loss plots)

Run:

```bash
uv sync --extra dev --extra grandfinals
uv run python scripts/generate_offline_corpus.py
uv run python train_adversarial.py
uv run python scripts/plot_training_metrics.py
```

This writes:
- `runs/training_metrics.jsonl`
- `plots/reward_curve.png`
- `plots/entropy_curve.png`

Embed those plots in your README.

## 7) What “training” means here (in this repo)

This repo is a **gym + grader + interface**. The live UI trainer is a **deterministic simulator** that:

- runs real episodes through `AgentGuardEnvironment`
- applies either a `heuristic` or `random` policy
- streams per-step rewards/outcomes in real time

If you later add RL fine-tuning (TRL/GRPO/etc.), you can wire your trained policy into the same
dashboard by replacing the action policy used by the trainer.

