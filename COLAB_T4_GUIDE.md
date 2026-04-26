# Colab (T4) guide — AgentGuard-Gym GRPO

Use this with the **public training notebook** (judge re-run):

- **Open in Google Colab:**  
  `https://colab.research.google.com/github/shekhar871/New-Project-A-and-B/blob/hf-split/notebooks/AgentGuard_GRPO_training.ipynb`  
  (If your default branch is `main` and the notebook lives there, replace `hf-split` with `main` in the URL.)

- **Same notebook in the Hugging Face Space repo (files):**  
  `https://huggingface.co/spaces/shekhar1090/agentguard-gym-final/blob/main/notebooks/AgentGuard_GRPO_training.ipynb`  
  (After you push the Space; path is `notebooks/AgentGuard_GRPO_training.ipynb`.)

## Why Colab first, HF Jobs later

- **Colab:** Fast iteration, free/cheap **T4**, first **loss + reward** plots for evidence.
- **HF Jobs:** Stable GPU log + artifact links for a “final” run.

## T4-safe defaults (see `train_adversarial.py`)

- `NUM_GENERATIONS=4`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`
- `MAX_COMPLETION_LENGTH=128` (override via env if needed)
- Smaller model if 7B OOM: `export MODEL_HF=Qwen/Qwen2.5-0.5B-Instruct`

## Public re-run without W&B login

The training script respects:

```bash
export WANDB_MODE=disabled
export WANDB_DISABLED=true
```

Metrics still append to **`runs/training_metrics.jsonl`** via `JsonlTrainingMetricsCallback`.

## Manual steps (same as the notebook)

From the **repository root** (not a subfolder):

```bash
git clone -b hf-split https://github.com/shekhar871/New-Project-A-and-B.git
cd New-Project-A-and-B
uv sync --extra dev --extra grandfinals
# or: pip install -e ".[dev,grandfinals]"
```

```bash
uv run python scripts/generate_offline_corpus.py
uv run python train_adversarial.py
uv run python scripts/plot_training_metrics.py
```

**Evidence files (judges):**

- `runs/training_metrics.jsonl` — one JSON line per step (`loss`, `entropy`, `win_rate`, `fp_rate`)
- `results/training_curve.png` — **loss** (if logged) + outcome rates + entropy

## Verify (local or Colab)

```bash
uv run pytest
uv run openenv validate --verbose .
uv run server
# open http://127.0.0.1:7860/ui
```

Dashboard port is **7860** (OpenEnv / Hugging Face Spaces default in this project), not 8000.
