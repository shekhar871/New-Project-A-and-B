# train_adversarial.py — complete, runnable training script

import os, json
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
import wandb

from agentguard_gym.training.reward_fns import defender_reward_fn
from agentguard_gym.training.callbacks import EntropyGuardCallback

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
OUTPUT_DIR = "./checkpoints/defender"
WANDB_PROJECT = "agentguard-adversarial"
CORPUS_PATH = "data/offline_corpus.jsonl"
# ────────────────────────────────────────────────────────

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

def main():
    wandb.init(project=WANDB_PROJECT, name="defender-grpo-v2")
    
    # Load model (4-bit QLoRA)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    # GRPO config with research-backed hyperparameters
    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,      # Effective batch = 8
        learning_rate=5e-6,
        beta=0.04,                           # Initial KL; annealed by callback
        num_generations=4,                   # G=4 (OOM safe on T4/V100)
        max_completion_length=200,           # Short = more steps = better curves
        logging_steps=1,
        save_steps=25,
        report_to="wandb",
        remove_unused_columns=False,
        # DAPO Clip-Higher: asymmetric clipping (prevents entropy collapse)
        # If TRL version supports it:
        # cliprange_high=0.28,
        # cliprange_low=0.20,
    )
    
    dataset = load_dataset(CORPUS_PATH)
    
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
    model.save_pretrained(f"{OUTPUT_DIR}/adapter_final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter_final")
    print(f"Saved adapter to {OUTPUT_DIR}/adapter_final")
    
    wandb.finish()

if __name__ == "__main__":
    main()
