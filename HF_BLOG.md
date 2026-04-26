# AgentGuard-Gym — Grand Finals writeup (HF blog)

This is the “mini-blog” writeup requested by judges. It is intentionally **separate from the README** and is committed directly into the Hugging Face Space repo.

## What we built

**AgentGuard-Gym** is an OpenEnv-style cybersecurity environment + Grand Finals adversarial RL stack for **agentic AI defense**. It targets three SOC-style threat themes aligned with **OWASP Agentic AI (2026)** framing:

- **Prompt injection** (easy)
- **Tool-chain misuse / SSRF** (medium)
- **Memory poisoning / privilege escalation** (hard)

The same codebase provides:

- A **deterministic grader-based gym** (fast to validate, reproducible scoring).
- A **Grand Finals adversarial system**: attacker → defender → judge, curriculum, ELO, novelty, and an RL training entrypoint (GRPO).
- A **real-time dashboard** on the Space (`/ui`) driven by **Server-Sent Events** (`/events`), showing episodes and simulated training telemetry.

## Why it’s credible (not a toy)

### 1) Realistic success metric: confusion-matrix utility
The environment maps defender actions to **TP/TN/FP/FN** outcomes and shapes rewards using bounded utility + time potential terms (MTTD/MTTR-style). Rewards are normalized to **[0, 1]** so dashboards and RL do not silently diverge across tasks.

### 2) “Benign traffic” is first-class
A defender that always blocks is useless in practice. Grand Finals adversarial episodes therefore include **~30% benign scenarios** sampled from `data/benign_corpus.json`, so the system explicitly measures **false positives** and trains against collapse-to-block.

### 3) Hybrid judging (continuous)
The judge path supports a rules-based scorer that yields a **continuous 0–1 quality signal** (not just discrete buckets). This provides a smooth attacker “sophistication” scalar even when LLM judging is disabled.

### 4) Novelty scoring for attacker pressure
Attack novelty is computed via **SentenceTransformers cosine similarity** when available (with caching) and a portable fallback when not. This discourages repeated attacks and gives the attacker a meaningful diversity incentive.

## How it works (high level)

### Core gym (OpenEnv API)

- `POST /reset` → begin an episode
- `POST /step` → submit a structured action (e.g. `allow`, `block`, `audit_tool_chain`, `quarantine_memory`, …)
- returns a structured observation and a reward in **[0,1]**

### Grand Finals adversarial loop

For each episode:

1. Sample **benign** (TN/FP) with probability ~0.30, else create a **malicious** scenario (TP/FN).
2. Build a defender observation (narrative + artifacts + tool trace).
3. The defender responds (heuristic baseline in the Space UI; RL/LLM options exist for training/offline runs).
4. Judge scores the interaction and the system updates:
   - **ELO**
   - **curriculum difficulty**
   - **novelty statistics**

## Training story (what to run)

The Space is for **environment + UI**. Training is run separately (Colab T4 / HF Jobs):

- See the Colab/T4 runbook in `COLAB_T4_GUIDE.md`.
- The training entrypoint is `train_adversarial.py` (GRPO-style, Unsloth optional, fallback to transformers+PEFT+4bit).

## How to verify (what judges can check quickly)

- Space health: `GET /health`
- UI: `GET /ui`
- One episode trace: `POST /episode/run`
- One adversarial episode: `POST /adversarial/episode`
- Real-time events (SSE): `GET /events`

## Links

- **Space**: https://huggingface.co/spaces/shekhar1090/agentguard-gym-final
- **Colab/T4 runbook**: `COLAB_T4_GUIDE.md`
- **YouTube demo (optional)**: add your link in `README.md` under “Submission links”.

