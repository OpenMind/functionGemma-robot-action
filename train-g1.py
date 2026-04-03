"""
FunctionGemma 270M Fine-tuning for Robot Action + Emotion Control
Based on: https://ai.google.dev/gemma/docs/functiongemma/finetuning-with-functiongemma
          https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices
"""

import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# CONFIG
BASE_MODEL = "google/functiongemma-270m-it"
TRAIN_FILE = "train-g1.jsonl"  # Your training data
OUTPUT_DIR = "./functiongemma-robot-actions"
EPOCHS = 5
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512


# FUNCTION DEFINITIONS (shared across all examples)
# These define your robot's API surface
FUNCTION_DEFINITIONS = [
    {
        "name": "robot_action",
        "description": "Execute a predefined robot action or gesture",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action_name": {
                    "type": "STRING",
                    "description": "The action to perform",
                    "enum": [
                        "shake_hand",
                        "face_wave",
                        "hands_up",
                        "stand_still",
                        "show_hand",
                    ],
                }
            },
            "required": ["action_name"],
        },
    },
    {
        "name": "show_emotion",
        "description": "Display an emotion on the robot avatar screen using Rive animations",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "emotion": {
                    "type": "STRING",
                    "description": "The emotion to display",
                    "enum": ["happy", "sad", "excited", "confused", "curious", "think"],
                }
            },
            "required": ["emotion"],
        },
    },
]


def build_function_declaration_str(func_def: dict) -> str:
    """
    Build FunctionGemma function declaration string.
    Format: <start_function_declaration>declaration:name{description:...,parameters:{...}}<end_function_declaration>
    Reference: https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices
    """
    name = func_def["name"]
    desc = func_def["description"]
    params = json.dumps(func_def["parameters"])

    return (
        f"<start_function_declaration>"
        f"declaration:{name}"
        f"{{description:<escape>{desc}<escape>,parameters:{params}}}"
        f"<end_function_declaration>"
    )


def build_function_call_str(tool_name: str, tool_args: dict) -> str:
    """
    Build FunctionGemma function call string.
    Format: <start_function_call>call:name{key:value,...}<end_function_call>
    """
    args_parts = []
    for key, value in tool_args.items():
        if value is None:
            args_parts.append(f"{key}:None")
        else:
            args_parts.append(f"{key}:<escape>{value}<escape>")

    args_str = ",".join(args_parts)
    return f"<start_function_call>call:{tool_name}{{{args_str}}}<end_function_call>"


def build_system_message() -> str:
    """Build the developer/system message with all function declarations."""
    declarations = "".join(
        build_function_declaration_str(f) for f in FUNCTION_DEFINITIONS
    )
    return (
        "You are a robot action controller. "
        "When the user gives a command, call the appropriate functions. "
        "Always call both a robot_action AND show_emotion together.\n" + declarations
    )


def format_training_example(example: dict) -> str:
    """
    Convert a training example to FunctionGemma chat format.

    FunctionGemma format:
    <bos><start_of_turn>developer\n{system_msg}<end_of_turn>
    <start_of_turn>user\n{user_prompt}<end_of_turn>
    <start_of_turn>model\n{function_calls}<end_of_turn><eos>
    """
    system_msg = build_system_message()
    user_prompt = example["user_prompt"]

    # Build function call string(s)
    tool_calls = example["tool_calls"]
    calls_str = ""
    for call in tool_calls:
        calls_str += build_function_call_str(call["tool_name"], call["tool_args"])

    formatted = (
        f"<bos><start_of_turn>developer\n{system_msg}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n{calls_str}<end_of_turn><eos>"
    )
    return formatted


def load_and_format_dataset(filepath: str) -> Dataset:
    """Load JSONL file and format for training."""
    examples = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            formatted_text = format_training_example(example)
            examples.append({"text": formatted_text})

    dataset = Dataset.from_list(examples)
    print(f"Loaded {len(dataset)} training examples")
    print("\n--- Example formatted text ---")
    print(dataset[0]["text"][:500])
    print("...")
    return dataset


def main():
    print("=" * 60)
    print("FunctionGemma 270M Fine-tuning for Robot Actions")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected, training will be very slow!")

    # Load dataset
    print("\n[1/5] Loading dataset...")
    dataset = load_and_format_dataset(TRAIN_FILE)

    # Split 90/10 for train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Load model and tokenizer
    print("\n[2/5] Loading FunctionGemma 270M...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    tokenizer.truncation_side = "left"

    # Fix vocab size mismatch (base model vs tokenizer)
    # This ensures the merged model exports correctly without needing remerge.py
    model.resize_token_embeddings(len(tokenizer))

    print(f"Model parameters: {model.num_parameters() / 1e6:.1f}M")

    # Configure LoRA
    print("\n[3/5] Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training arguments
    print("\n[4/5] Setting up trainer...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_accumulation_steps=2,
        dataloader_num_workers=2,
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )

    # Train
    print("\n[5/5] Starting training...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max seq length: {MAX_SEQ_LENGTH}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print()

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"Model saved to {OUTPUT_DIR}/final")

    # Merge LoRA weights into base model for deployment
    print("\nMerging LoRA weights...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(f"{OUTPUT_DIR}/merged")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged")
    print(f"Merged model saved to {OUTPUT_DIR}/merged")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  LoRA adapter: {OUTPUT_DIR}/final")
    print(f"  Merged model: {OUTPUT_DIR}/merged")
    print("=" * 60)
    print("\nNext step: Run eval.py to test the model")


if __name__ == "__main__":
    main()
