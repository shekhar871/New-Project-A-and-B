"""
train_adversarial.py — Grand Finals training script (defender-only GRPO).

T4-safe defaults:
- num_generations=4 (G=4)
- small batch + grad accumulation
- short completion length
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from agentguard_gym.training.callbacks import EntropyGuardCallback
from agentguard_gym.training.reward_fns import defender_reward_fn


@dataclass(frozen=True)
class TrainCfg:
    model_name: str = os.getenv("AG_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    output_dir: str = os.getenv("AG_OUTPUT_DIR", "./checkpoints/defender")
    corpus_path: str = os.getenv("AG_CORPUS_PATH", "data/offline_corpus.jsonl")
    use_wandb: bool = os.getenv("WANDB_DISABLED", "true").lower() not in ("1", "true", "yes")


def _maybe_init_wandb(enabled: bool) -> None:
    if not enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return
    import wandb  # type: ignore

    wandb.init(project=os.getenv("WANDB_PROJECT", "agentguard-adversarial"), name=os.getenv("WANDB_NAME", "defender-grpo"))

def load_dataset(path: str) -> Dataset:
    """Load pre-generated offline corpus as HuggingFace Dataset."""
    from agentguard_gym.grandfinals.observation_builder import build_defender_prompt
    
    records = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            prompt = build_defender_prompt(
                narrative=_build_narrative(item),
                task=item["task"],
                is_malicious=item.get("is_malicious_ground_truth", True),
            )
            records.append({"prompt": prompt})
    
    print(f"Loaded {len(records)} training examples from corpus")
    return Dataset.from_list(records)

def _build_narrative(item: dict) -> str:
    task = item.get("task", "")
    text = item.get("attack_text", "")
    if task == "prompt_injection":
        return f'A user message: "{text}"'
    elif task == "tool_misuse_ssrf":
        return f"Tool fetch_url called with: {text}"
    else:
        return f"Memory write: \"{text}\""

def _load_model_and_tokenizer(model_name: str):
    """
    Prefer Unsloth if available; otherwise fall back to Transformers+PEFT.
    This keeps Colab/HF-jobs reproducible even if unsloth isn't installed.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer
    except Exception:
        from peft import LoraConfig, get_peft_model  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        lora = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora)
        return model, tokenizer


def main() -> None:
    cfg = TrainCfg()
    _maybe_init_wandb(cfg.use_wandb)

    model, tokenizer = _load_model_and_tokenizer(cfg.model_name)
    
    # GRPO config with research-backed hyperparameters
    config = GRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=int(os.getenv("AG_EPOCHS", "1")),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(os.getenv("AG_GRAD_ACCUM", "8")),  # effective batch ~8
        learning_rate=float(os.getenv("AG_LR", "5e-6")),
        beta=0.04,                           # Initial KL; annealed by callback
        num_generations=4,                   # G=4 (OOM safe on T4/V100)
        max_completion_length=int(os.getenv("AG_MAX_COMPLETION", "200")),
        logging_steps=1,
        save_steps=int(os.getenv("AG_SAVE_STEPS", "25")),
        report_to="wandb",
        remove_unused_columns=False,
        # DAPO Clip-Higher: asymmetric clipping (prevents entropy collapse)
        # If TRL version supports it:
        # cliprange_high=0.28,
        # cliprange_low=0.20,
    )
    
    dataset = load_dataset(cfg.corpus_path)
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[defender_reward_fn],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[
            EntropyGuardCallback(beta_0=0.04, decay=0.5, t_max=200),
        ],
    )
    
    print("Starting GRPO training...")
    print(f"Dataset size: {len(dataset)}")
    print(f"G (responses per prompt): {config.num_generations}")
    print(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    
    trainer.train()
    
    # Save ONLY adapter (never merge — risk of weight corruption)
    final_dir = f"{cfg.output_dir}/adapter_final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved adapter to {final_dir}")

    try:
        import wandb  # type: ignore

        wandb.finish()
    except Exception:
        pass

if __name__ == "__main__":
    main()
