# train_adversarial.py — complete, runnable training script
# Works with or without Unsloth (transformers+PEFT+4bit fallback).

import json
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

try:
    from unsloth import FastLanguageModel

    USE_UNSLOTH = True
except ImportError:
    USE_UNSLOTH = False
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("[WARN] Unsloth not found — using transformers+PEFT+4bit fallback (install unsloth for best speed)")

# ── Config ──────────────────────────────────────────────
MODEL_UNSLOTH = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MODEL_HF = "Qwen/Qwen2.5-7B-Instruct"
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
    print(f"Loaded {len(records)} training examples from corpus (metadata: SHA->GT, not [GT:] tags)")
    return Dataset.from_list(records)


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

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_HF,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
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

    wandb.init(project=WANDB_PROJECT, name="defender-grpo-v2")

    sft = data_dir / "sft_warmstart.json"
    if sft.is_file():
        print(f"[info] SFT warmstart file present: {sft} — optional phase-1 SFT is project-specific; GRPO below is the default path.")

    model, tokenizer = _load_model_and_tokenizer()

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        beta=0.04,
        num_generations=4,
        max_completion_length=200,
        logging_steps=1,
        save_steps=25,
        report_to="wandb",
        remove_unused_columns=False,
    )

    dataset = load_grpo_corpus(CORPUS_PATH)

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
    print(f"  dataset size: {len(dataset)}")

    trainer.train()

    model.save_pretrained(f"{OUTPUT_DIR}/adapter_final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter_final")
    print(f"Saved adapter to {OUTPUT_DIR}/adapter_final")

    wandb.finish()


if __name__ == "__main__":
    main()
