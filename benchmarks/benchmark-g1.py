"""
Constrained Decoding: Instead of generating 33 tokens autoregressively,
force-feed the known template tokens and only let the model "decide"
at the ACTION and EMOTION positions.

Output format is always:
  <start_function_call>call:robot_action{action_name:<escape>ACTION<escape>}
  <end_function_call><start_function_call>call:show_emotion{emotion:<escape>EMOTION<escape>}
  <end_function_call><end_of_turn>

Of 33 tokens, only ~2 are actual decisions. This should cut decode from
580ms to ~60ms.
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Model source — works with both Hugging Face repo ID and local path
# ============================================================
HF_MODEL = "wenjinf0811/functiongemma-robot-actions"
LOCAL_MODEL = "./functionGemma-finetuned-g1"  # For Google Drive download

ACTIONS = ["shake_hand", "face_wave", "hands_up", "stand_still", "show_hand"]
EMOTIONS = ["happy", "sad", "excited", "confused", "curious", "think"]

FUNCTIONS = [
    {
        "name": "robot_action",
        "description": "Execute a predefined robot action or gesture",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action_name": {
                    "type": "STRING",
                    "description": "The action to perform",
                    "enum": ACTIONS,
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
                    "enum": EMOTIONS,
                }
            },
            "required": ["emotion"],
        },
    },
]


def build_prompt(user_input):
    decls = ""
    for f in FUNCTIONS:
        decls += (
            f"<start_function_declaration>declaration:{f['name']}"
            f"{{description:<escape>{f['description']}<escape>,"
            f"parameters:{json.dumps(f['parameters'])}}}"
            f"<end_function_declaration>"
        )
    system = (
        "You are a robot action controller. When the user gives a command, call the appropriate functions. Always call both a robot_action AND show_emotion together.\n"
        + decls
    )
    return f"<bos><start_of_turn>developer\n{system}<end_of_turn>\n<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"


def load_model(model_path):
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_constrained(
    model,
    input_ids,
    action_prefix,
    action_suffix,
    emotion_suffix,
    action_token_ids,
    emotion_token_ids,
    valid_action_first_tokens,
    valid_emotion_first_tokens,
):
    """
    Constrained generation with BATCHED token feeding.

    Total: ~2 forward passes instead of 33!
      1. Prefill (input + action_prefix) -> pick action
      2. Feed (action_tokens + action_suffix) -> pick emotion
    """
    device = input_ids.device

    # === Forward pass 1: Prefill input + action prefix ===
    prefix_ids = torch.tensor([action_prefix], dtype=torch.long, device=device)
    full_input = torch.cat([input_ids, prefix_ids], dim=1)
    outputs = model(input_ids=full_input, use_cache=True)
    past = outputs.past_key_values

    # Pick action (constrained to valid first tokens)
    logits = outputs.logits[:, -1, :]
    mask = torch.full_like(logits, float("-inf"))
    for tok_id in valid_action_first_tokens:
        mask[0, tok_id] = 0
    action_first_token = (logits + mask).argmax(dim=-1).item()

    # Find which action
    chosen_action = None
    for a, ids in action_token_ids.items():
        if ids[0] == action_first_token:
            chosen_action = a
            break

    # === Forward pass 2: Feed action tokens + suffix IN ONE BATCH ===
    action_ids = action_token_ids[chosen_action]
    combined = action_ids + action_suffix
    combined_tensor = torch.tensor([combined], dtype=torch.long, device=device)
    outputs = model(input_ids=combined_tensor, past_key_values=past, use_cache=True)
    past = outputs.past_key_values

    # Pick emotion (constrained)
    logits = outputs.logits[:, -1, :]
    mask = torch.full_like(logits, float("-inf"))
    for tok_id in valid_emotion_first_tokens:
        mask[0, tok_id] = 0
    emotion_first_token = (logits + mask).argmax(dim=-1).item()

    chosen_emotion = None
    for e, ids in emotion_token_ids.items():
        if ids[0] == emotion_first_token:
            chosen_emotion = e
            break

    return chosen_action, chosen_emotion


def main():
    parser = argparse.ArgumentParser(description="FunctionGemma Robot Benchmark")
    parser.add_argument(
        "--local", action="store_true", help="Use local model instead of Hugging Face"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Custom model path or HF repo ID"
    )
    args = parser.parse_args()

    if args.model:
        model_path = args.model
    elif args.local:
        model_path = LOCAL_MODEL
    else:
        model_path = HF_MODEL

    model, tokenizer = load_model(model_path)

    # Pre-tokenize all valid action and emotion values
    action_token_ids = {}
    for a in ACTIONS:
        action_token_ids[a] = tokenizer.encode(a, add_special_tokens=False)

    emotion_token_ids = {}
    for e in EMOTIONS:
        emotion_token_ids[e] = tokenizer.encode(e, add_special_tokens=False)

    # Pre-tokenize the template parts
    action_prefix = tokenizer.encode(
        "<start_function_call>call:robot_action{action_name:<escape>",
        add_special_tokens=False,
    )
    action_suffix = tokenizer.encode(
        "<escape>}<end_function_call><start_function_call>call:show_emotion{emotion:<escape>",
        add_special_tokens=False,
    )
    emotion_suffix = tokenizer.encode(
        "<escape>}<end_function_call><end_of_turn>",
        add_special_tokens=False,
    )

    print(f"Action prefix tokens: {len(action_prefix)}")
    print(f"Action suffix tokens: {len(action_suffix)}")
    print(f"Emotion suffix tokens: {len(emotion_suffix)}")
    print()

    print("Action token mappings:")
    for a, ids in action_token_ids.items():
        print(f"  {a:<20s} -> {ids} ({len(ids)} tokens)")
    print()
    print("Emotion token mappings:")
    for e, ids in emotion_token_ids.items():
        print(f"  {e:<20s} -> {ids} ({len(ids)} tokens)")
    print()

    # Build constrained token sets
    valid_action_first_tokens = {ids[0] for ids in action_token_ids.values()}
    valid_emotion_first_tokens = {ids[0] for ids in emotion_token_ids.values()}

    # Warmup
    print("Warming up...")
    for _ in range(3):
        inputs = tokenizer(build_prompt("hello"), return_tensors="pt").to(model.device)
        generate_constrained(
            model,
            inputs["input_ids"],
            action_prefix,
            action_suffix,
            emotion_suffix,
            action_token_ids,
            emotion_token_ids,
            valid_action_first_tokens,
            valid_emotion_first_tokens,
        )
    print("Done!\n")

    # Benchmark
    tests = [
        "Shake hands!",
        "Wave at me",
        "Put your hands up",
        "I feel sad",
        "Show me your hand",
        "Just stand there",
        "Good boy!",
        "What is that?",
        "Hello there!",
        "You are cute!",
        "Tell me a joke",
        "How are you today?",
        "What's the weather like?",
        "Nice to meet you",
        "Thank you so much",
    ]

    print("Running benchmark...\n")
    times = []
    for t in tests:
        inputs = tokenizer(build_prompt(t), return_tensors="pt").to(model.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        action, emotion = generate_constrained(
            model,
            inputs["input_ids"],
            action_prefix,
            action_suffix,
            emotion_suffix,
            action_token_ids,
            emotion_token_ids,
            valid_action_first_tokens,
            valid_emotion_first_tokens,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = (time.time() - start) * 1000

        times.append(ms)
        print(f"{ms:6.0f}ms  {t:<30s} action={action:<16s} emotion={emotion}")

    print("\n--- Results ---")
    print(f"Min:     {min(times):.0f}ms")
    print(f"Max:     {max(times):.0f}ms")
    print(f"Average: {sum(times)/len(times):.0f}ms")


if __name__ == "__main__":
    main()
