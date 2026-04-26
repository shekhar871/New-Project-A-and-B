# AgentGuard-Gym — mini-blog (Hugging Face)

This file is the **short write-up** for judges: what the environment does, what we trained, and where to run things. **Do not host large videos in the Hub repo** — use public YouTube links from `README.md` only.

## What we built

**AgentGuard-Gym** is an [OpenEnv](https://github.com/openenv/)-style cybersecurity **gym** plus a **Grand Finals** adversarial stack for **agentic AI defense**. It targets three SOC-style themes aligned with **OWASP Agentic AI (2026)** framing:

- **Prompt injection** (easy)
- **Tool-chain misuse / SSRF** (medium)
- **Memory poisoning / privilege escalation** (hard)

The repo delivers:

1. A **deterministic, grader-based** `reset` / `step` environment with rewards in **[0, 1]**.
2. An **adversarial loop**: attacker → defender → judge, with **~30% benign** traffic (TN/FP) from `data/benign_corpus.json` so the policy cannot collapse to “always block”.
3. **GRPO training** via **Hugging Face TRL** (`train_adversarial.py`), optional **Unsloth**, with **experience replay** mixing (`training/replay.py`, `REPLAY_RATIO`).
4. A **FastAPI** server and **live dashboard** (`/ui`, SSE `/events`) for episodes and trainer metrics.

## What we trained

We run **GRPO** on an **offline defender prompt corpus** (`data/offline_corpus.jsonl` from `scripts/generate_offline_corpus.py`) with a **strict SHA-256 ground-truth registry** in `agentguard_gym/training/reward_fns.py` (rewards do not depend on fragile `[GT:…]` substrings in prompts).

**Training evidence** from a real run (required for competition):

- Per-step log: `runs/training_metrics.jsonl` (e.g. `loss`, `entropy`, `win_rate`, `fp_rate`)
- Plots: `uv run python scripts/plot_training_metrics.py` → **`results/training_curve.png`**

**Public re-run without W&B login** (e.g. Colab): set `WANDB_MODE=disabled` and `WANDB_DISABLED=true`; JSONL + plots still work.

## Judge links (one place)

| Item | URL |
|------|-----|
| **Run the environment (Space)** | [Hugging Face Space](https://huggingface.co/spaces/shekhar1090/agentguard-gym-final) |
| **Training notebook (Colab)** | [Open in Colab](https://colab.research.google.com/github/shekhar871/New-Project-A-and-B/blob/hf-split/notebooks/AgentGuard_GRPO_training.ipynb) |
| **T4 runbook** | [COLAB_T4_GUIDE.md in Space](https://huggingface.co/spaces/shekhar1090/agentguard-gym-final/blob/main/COLAB_T4_GUIDE.md) |
| **This blog (MD)** | [HF_BLOG.md in Space](https://huggingface.co/spaces/shekhar1090/agentguard-gym-final/blob/main/HF_BLOG.md) |
| **Main README (all references)** | [README in Space](https://huggingface.co/spaces/shekhar1090/agentguard-gym-final/blob/main/README.md) |
| **YouTube (optional)** | See *Submission links* in `README.md` when published |

## How to verify quickly

- `GET /health` — liveness
- `GET /ui` — dashboard
- `POST /episode/run` — one full episode with trace
- `POST /adversarial/episode` — one adversarial episode
- `GET /events` — SSE stream

## License

`BSD-3-Clause` (see `pyproject.toml`).
