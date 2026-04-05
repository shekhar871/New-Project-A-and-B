# Deploying to Hugging Face (Round 1)

## Why `openenv push` failed with your current tokens

Fine-grained tokens that only grant **Inference / serverless write** can call the **router API** (`inference.py`) but **cannot** create or update **Spaces** or **model repos**.

You need **one** of:

1. A **classic** token with **Write** access, or  
2. A **fine-grained** token that includes **Repositories: write** and **Spaces: write** (or equivalent) for your user namespace.

Then:

```bash
export HF_TOKEN="hf_..."   # token with Space write
cd agentguard-gym
uv run openenv push -r YOUR_USERNAME/agentguard-gym-openenv
# If `openenv` is on your PATH globally, you can use `openenv push` instead.
```

Repeat for `aml-defense-gym` with a **second Space** name.

## Manual Space (if you prefer the UI)

1. New Space → Docker → empty.  
2. Push this repo (or connect GitHub).  
3. Ensure the Space Dockerfile matches the root `Dockerfile` here.  
4. Tag the Space with **`openenv`** per hackathon rules.

## Security

**Rotate any token that was pasted into chat, email, or Cursor** — treat it as compromised.
