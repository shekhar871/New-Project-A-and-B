# Deploying to Hugging Face (Round 1)

Same constraints as **AgentGuard-Gym**: `openenv push` needs a token with **Spaces** (and repo) **write** rights, not inference-only.

```bash
export HF_TOKEN="hf_..."   # write-capable token
cd aml-defense-gym
uv run openenv push -r YOUR_USERNAME/aml-defense-gym-openenv
# If `openenv` is on your PATH globally, you can use `openenv push` instead.
```

See `agentguard-gym/DEPLOYMENT.md` for details and security notes.
