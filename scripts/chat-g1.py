import argparse
import json
import re
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
        decls += f"<start_function_declaration>declaration:{f['name']}{{description:<escape>{f['description']}<escape>,parameters:{json.dumps(f['parameters'])}}}<end_function_declaration>"
    system = (
        "You are a robot action controller. When the user gives a command, call the appropriate functions. Always call both a robot_action AND show_emotion together.\n"
        + decls
    )
    return f"<bos><start_of_turn>developer\n{system}<end_of_turn>\n<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"


def parse_calls(text):
    calls = []
    for m in re.finditer(
        r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>", text
    ):
        args = {}
        for part in m.group(2).split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                args[k.strip()] = v.replace("<escape>", "").strip()
        calls.append({"function": m.group(1), "args": args})
    return calls


def load_model(model_path):
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Device: {model.device}")
    print("Ready!\n")
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FunctionGemma Robot Chat")
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

    while True:
        user = input("You: ").strip()
        if user.lower() in ("quit", "exit", "q"):
            break
        inputs = tokenizer(build_prompt(user), return_tensors="pt").to(model.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = (time.time() - start) * 1000

        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        calls = parse_calls(generated)
        print(f"Robot ({ms:.0f}ms): {json.dumps(calls, indent=2)}\n")
