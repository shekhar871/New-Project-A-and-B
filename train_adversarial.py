# train_adversarial.py — complete, runnable training script
# Works with or without Unsloth (transformers+PEFT+4bit fallback).

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import wandb

from agentguard_gym.training.callbacks import (
    EntropyGuardCallback,
    JsonlTrainingMetricsCallback,
    WandbOutcomeMetricsCallback,
)
from agentguard_gym.training.reward_fns import (
    clear_training_prompt_metadata,
    defender_reward_fn,
    register_training_prompt_metadata,
    reset_outcome_tally,
)
from agentguard_gym.training.replay import ReplayMixedPrompts

try:
    from unsloth import FastLanguageModel

    USE_UNSLOTH = True
except ImportError:
    USE_UNSLOTH = False
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("[WARN] Unsloth not found — using transformers+PEFT+4bit fallback (install unsloth for best speed)")

# ── Config (override via env) ────────────────────────────
MODEL_UNSLOTH = str(os.getenv("MODEL_UNSLOTH", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"))
MODEL_HF = str(os.getenv("MODEL_HF", "Qwen/Qwen2.5-7B-Instruct"))
OUTPUT_DIR = "./checkpoints/defender"
WANDB_PROJECT = "agentguard-adversarial"
CORPUS_PATH = "data/offline_corpus.jsonl"
# ────────────────────────────────────────────────────────


def load_grpo_corpus(path: str) -> Dataset:
    """Load pre-generated offline corpus; ground truth registered by prompt hash (not in prompt)."""
    from agentguard_gym.grandfinals.observation_builder import build_defender_prompt

    clear_training_prompt_metadata()
    records: list = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            is_mal = item.get("is_malicious_ground_truth", True)
            task = str(item.get("task", "prompt_injection"))
            prompt = build_defender_prompt(
                narrative=_build_narrative(item),
                task=task,
                is_malicious=is_mal,
                include_ground_truth_tags=False,
            )
            register_training_prompt_metadata(prompt=prompt, is_malicious=is_mal, task=task)
            records.append({"prompt": prompt})
    assert len(records) > 0, "GRPO corpus is empty."
    # Registry safety: strict reward_fn requires metadata
    from agentguard_gym.training.reward_fns import PROMPT_SHA_TO_META

    assert len(PROMPT_SHA_TO_META) > 0, "Prompt metadata registry empty — register_training_prompt_metadata was not called."
    print(f"Loaded {len(records)} training examples from corpus (metadata: SHA->GT, not [GT:] tags)")
    return Dataset.from_list(records)


def load_sft_warmstart(path: str) -> Dataset:
    """
    Minimal SFT warmstart dataset:
    - preferred: JSON list of {"prompt": "...", "completion": "..."} (gold pairs)
    - backward compatible: {"task","narrative","response_json"} (older format)
    Produces rows with `prompt` and `completion` for TRL SFTTrainer.
    """
    from agentguard_gym.grandfinals.observation_builder import build_defender_prompt

    p = Path(path)
    if not p.is_file():
        return Dataset.from_list([])
    rows = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        if "prompt" in r and "completion" in r:
            out.append({"prompt": str(r["prompt"]), "completion": str(r["completion"])})
            continue
        task = str(r.get("task", "prompt_injection"))
        narrative = str(r.get("narrative", "")).strip()
        completion_obj = r.get("response_json") or {}
        prompt = build_defender_prompt(
            narrative=narrative,
            task=task,
            is_malicious=True,
            include_ground_truth_tags=False,
        )
        out.append({"prompt": prompt, "completion": json.dumps(completion_obj, ensure_ascii=False)})
    return Dataset.from_list(out)


def _build_narrative(item: dict) -> str:
    task = item.get("task", "")
    text = item.get("attack_text", "")
    if task == "prompt_injection":
        return f'A user message: "{text}"'
    if task == "tool_misuse_ssrf":
        return f"Tool fetch_url called with: {text}"
    return f'Memory write: "{text}"'


def _load_model_and_tokenizer():
    if USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_UNSLOTH,
            max_seq_length=1024,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer

    # T4-safe defaults: use fp16 on CUDA unless explicitly overridden.
    # (T4 generally does not support bf16; fp16 is the correct default.)
    if torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    use_4bit = bool(torch.cuda.is_available())
    bnb = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        if use_4bit
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_HF,
        quantization_config=bnb,
        device_map=("auto" if use_4bit else None),
        trust_remote_code=True,
        torch_dtype=(torch.float16 if use_4bit else None),
    )
    peft = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    model.config.use_cache = False
    return model, tokenizer


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "training_started_at").touch()
    reset_outcome_tally()

    # ---------------------------------------------------------------------
    # PRODUCTION SAFETY (GRPO):
    # Reward metadata registry MUST be populated *before the first backward pass*.
    # If the registry is empty, rewards collapse to a constant -> advantage==0 -> zero gradient.
    # We load corpus + register metadata BEFORE constructing the trainer (and even before model load)
    # so any failure is caught early with explicit assertions.
    # ---------------------------------------------------------------------
    base_dataset = load_grpo_corpus(CORPUS_PATH)
    from agentguard_gym.training.reward_fns import PROMPT_SHA_TO_META

    assert len(PROMPT_SHA_TO_META) == len(base_dataset), (
        f"Registry has {len(PROMPT_SHA_TO_META)}/{len(base_dataset)} entries. "
        "Likely cause: prompt hashing mismatch or corpus encoding issues."
    )
    # Sentinel reward probe (“smoke test before the gun fires”)
    _probe_prompt = base_dataset[0]["prompt"]
    _probe_completion = "{\"decision\":\"block\"}"
    _probe_reward = defender_reward_fn([_probe_prompt], [_probe_completion])[0]
    assert (not isinstance(_probe_reward, float)) or (_probe_reward == _probe_reward), "Sentinel reward is NaN."
    assert float(_probe_reward) != 0.0, "Sentinel reward returned 0.0 — registry lookup/reward parsing likely failed."

    wandb.init(project=WANDB_PROJECT, name="defender-grpo-v2")
    try:
        wandb.log(
            {
                "setup/registry_size": float(len(PROMPT_SHA_TO_META)),
                "setup/sentinel_reward": float(_probe_reward),
            },
            step=0,
        )
    except Exception:
        pass

    # Track 1: Experience Replay mix-in for GRPO prompt sampling.
    # 30% replayed prompts, 70% fresh prompts (default matches BENIGN_EPISODE_PROB philosophy).
    replay_ratio = float(os.getenv("REPLAY_RATIO", "0.30"))
    dataset = ReplayMixedPrompts(base_prompts=list(base_dataset), replay_ratio=replay_ratio, prioritized=True)

    sft = data_dir / "sft_warmstart.json"
    if sft.is_file():
        print(
            f"[info] SFT warmstart file present: {sft} — optional phase-1 SFT is project-specific; GRPO below is the default path."
        )

    model, tokenizer = _load_model_and_tokenizer()

    # Optional SFT warmstart (G6): only runs if file has examples.
    sft_ds = load_sft_warmstart(str(sft))
    if len(sft_ds) > 0:
        try:
            from trl import SFTConfig, SFTTrainer

            print(f"[info] Running SFT warmstart on {len(sft_ds)} examples…")
            sft_trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=sft_ds,
                args=SFTConfig(
                    output_dir=str(Path(OUTPUT_DIR) / "sft"),
                    num_train_epochs=1,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=4,
                    learning_rate=1e-5,
                    logging_steps=1,
                    report_to=[],
                ),
            )
            sft_trainer.train()
        except Exception as e:
            print(f"[warn] SFT warmstart skipped due to error: {e}")

    # HF Jobs / Colab knobs (T4 defaults):
    # - MAX_STEPS: set to ~100 for quick evidence
    # - NUM_GENERATIONS: 4 is typical; reduce to 2 if OOM
    # - MAX_COMPLETION_LENGTH: keep short on T4
    max_steps = int(os.getenv("MAX_STEPS", "100"))
    num_generations = int(os.getenv("NUM_GENERATIONS", "4"))
    max_completion_length = int(os.getenv("MAX_COMPLETION_LENGTH", "128"))

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        max_steps=max_steps,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        beta=0.04,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        logging_steps=1,
        save_steps=25,
        report_to="wandb",
        fp16=bool(torch.cuda.is_available()),
        bf16=False,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[defender_reward_fn],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[
            EntropyGuardCallback(beta_0=0.04, decay=0.5, t_max=200),
            WandbOutcomeMetricsCallback(),
            JsonlTrainingMetricsCallback(),
        ],
    )

    print("Starting GRPO training...")
    print(f"  backend: {'Unsloth' if USE_UNSLOTH else 'transformers+PEFT+4bit'}")
    print(f"  dataset size: {len(dataset)} (replay_ratio={replay_ratio})")

    trainer.train()

    model.save_pretrained(f"{OUTPUT_DIR}/adapter_final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter_final")
    print(f"Saved adapter to {OUTPUT_DIR}/adapter_final")

    wandb.finish()


if __name__ == "__main__":
    main()
