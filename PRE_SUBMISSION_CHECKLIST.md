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

## 4) Run the server + UI (real-time episode/training view)

```bash
uv run server
```

Then open:

- `http://127.0.0.1:8000/ui` (dashboard)
- `http://127.0.0.1:8000/docs` (API)

On the dashboard:

- Click **Start trainer** to see episodes and rewards live.
- Click **Run one episode** to inspect step-by-step outcomes.

## 5) Baseline inference (LLM) — optional

Requires a token:

```bash
export HF_TOKEN=...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
uv run python inference.py
```

## 6) What “training” means here (in this repo)

This repo is a **gym + grader + interface**. The live UI trainer is a **deterministic simulator** that:

- runs real episodes through `AgentGuardEnvironment`
- applies either a `heuristic` or `random` policy
- streams per-step rewards/outcomes in real time

If you later add RL fine-tuning (TRL/GRPO/etc.), you can wire your trained policy into the same
dashboard by replacing the action policy used by the trainer.

