"""
FunctionGemma robot-action microservice for NVIDIA Jetson AGX Thor.
Run: uvicorn server:app --host 0.0.0.0 --port 8200.

Uses constrained decoding (2 forward passes instead of 33 autoregressive
steps) for ~59 ms average latency on Thor.

Endpoints
---------
POST /predict
    Convert natural language to robot action + emotion function calls.
POST /predict_batch
    Batch prediction for multiple inputs.
GET  /actions
    List supported actions and emotions.
GET  /health
    Health check.
"""

import json
import logging
import time

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("functiongemma-service")

app = FastAPI(title="FunctionGemma Robot Actions")
model = None
tokenizer = None

# Action / Emotion definitions — must match training data
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

# Constrained decoding state — populated at startup
action_prefix: list[int] = []
action_suffix: list[int] = []
action_token_ids: dict[str, list[int]] = {}
emotion_token_ids: dict[str, list[int]] = {}
valid_action_first_tokens: set[int] = set()
valid_emotion_first_tokens: set[int] = set()


# Request/Response Models
class PredictRequest(BaseModel):
    """
    Request body for single prediction.

    Attributes
    ----------
    text : str
        Natural language command or conversation input.
    """

    text: str


class PredictResponse(BaseModel):
    """
    Response body for single prediction.

    Attributes
    ----------
    action : str
        Robot action to perform (e.g. ``"shake_hand"``).
    emotion : str
        Emotion to display on avatar screen (e.g. ``"happy"``).
    latency_ms : float
        Model inference latency in milliseconds (excludes network I/O).
    """

    action: str
    emotion: str
    latency_ms: float


class BatchPredictRequest(BaseModel):
    """
    Request body for batch prediction.

    Attributes
    ----------
    texts : list of str
        List of natural language inputs to process.
    """

    texts: list[str]


class BatchPredictResponse(BaseModel):
    """
    Response body for batch prediction.

    Attributes
    ----------
    results : list of PredictResponse
        One result per input text.
    count : int
        Number of results returned.
    total_latency_ms : float
        Total inference latency in milliseconds.
    """

    results: list[PredictResponse]
    count: int
    total_latency_ms: float


def build_prompt(user_input: str) -> str:
    """Build FunctionGemma chat-format prompt with function declarations."""
    decls = ""
    for f in FUNCTIONS:
        decls += (
            f"<start_function_declaration>declaration:{f['name']}"
            f"{{description:<escape>{f['description']}<escape>,"
            f"parameters:{json.dumps(f['parameters'])}}}"
            f"<end_function_declaration>"
        )
    system = (
        "You are a robot action controller. When the user gives a command, "
        "call the appropriate functions. Always call both a robot_action AND "
        "show_emotion together.\n" + decls
    )
    return (
        f"<bos><start_of_turn>developer\n{system}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_input}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


@torch.no_grad()
def generate_constrained(input_ids: torch.Tensor) -> tuple[str, str]:
    """
    Constrained decoding — 2 forward passes instead of 33.

    Feeds known template tokens in bulk, only lets the model decide
    at ACTION and EMOTION positions. ~59 ms on Thor vs ~1085 ms
    with ``model.generate()``.

    Parameters
    ----------
    input_ids : torch.Tensor
        Tokenized prompt tensor of shape ``(1, seq_len)``.

    Returns
    -------
    tuple of (str, str)
        ``(chosen_action, chosen_emotion)``.
    """
    device = input_ids.device

    # Forward pass 1: prefill + pick action
    prefix_ids = torch.tensor([action_prefix], dtype=torch.long, device=device)
    full_input = torch.cat([input_ids, prefix_ids], dim=1)
    outputs = model(input_ids=full_input, use_cache=True)
    past = outputs.past_key_values

    logits = outputs.logits[:, -1, :]
    mask = torch.full_like(logits, float("-inf"))
    for tok_id in valid_action_first_tokens:
        mask[0, tok_id] = 0
    action_first_token = (logits + mask).argmax(dim=-1).item()

    chosen_action = None
    for a, ids in action_token_ids.items():
        if ids[0] == action_first_token:
            chosen_action = a
            break

    # Forward pass 2: feed action tokens + suffix, pick emotion
    combined = action_token_ids[chosen_action] + action_suffix
    combined_tensor = torch.tensor([combined], dtype=torch.long, device=device)
    outputs = model(
        input_ids=combined_tensor, past_key_values=past, use_cache=True
    )

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


@app.on_event("startup")
def load_model():
    """
    Load the FunctionGemma model onto GPU and pre-tokenize constrained
    decoding templates. Warmup ensures CUDA kernels are compiled before
    the first real request, avoiding cold-start latency (~2s -> ~59ms).
    """
    global model, tokenizer
    global action_prefix, action_suffix
    global action_token_ids, emotion_token_ids
    global valid_action_first_tokens, valid_emotion_first_tokens

    logger.info("Loading FunctionGemma on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained("OpenmindAGI/functiongemma-finetuned-g1")
    model = AutoModelForCausalLM.from_pretrained(
        "OpenmindAGI/functiongemma-finetuned-g1",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Pre-tokenize action/emotion values
    for a in ACTIONS:
        action_token_ids[a] = tokenizer.encode(a, add_special_tokens=False)
    for e in EMOTIONS:
        emotion_token_ids[e] = tokenizer.encode(e, add_special_tokens=False)

    # Pre-tokenize template segments
    action_prefix = tokenizer.encode(
        "<start_function_call>call:robot_action{action_name:<escape>",
        add_special_tokens=False,
    )
    action_suffix = tokenizer.encode(
        "<escape>}<end_function_call>"
        "<start_function_call>call:show_emotion{emotion:<escape>",
        add_special_tokens=False,
    )

    # Build constrained token sets
    valid_action_first_tokens.update(ids[0] for ids in action_token_ids.values())
    valid_emotion_first_tokens.update(ids[0] for ids in emotion_token_ids.values())

    # Warmup
    for _ in range(5):
        inputs = tokenizer(build_prompt("hello"), return_tensors="pt").to(model.device)
        generate_constrained(inputs["input_ids"])
    logger.info("Model ready!")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Convert natural language input to action + emotion.

    Parameters
    ----------
    req : PredictRequest
        Request body containing the text input.

    Returns
    -------
    PredictResponse
        Chosen action, emotion, and inference latency.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    inputs = tokenizer(build_prompt(req.text), return_tensors="pt").to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    action, emotion = generate_constrained(inputs["input_ids"])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = (time.perf_counter() - start) * 1000

    logger.info(
        'predict | text="%s" | action=%s emotion=%s | %.0fms',
        req.text[:50],
        action,
        emotion,
        latency,
    )
    return PredictResponse(
        action=action, emotion=emotion, latency_ms=round(latency, 1)
    )


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    """
    Process multiple inputs with constrained decoding.

    Parameters
    ----------
    req : BatchPredictRequest
        Request body containing list of text inputs.

    Returns
    -------
    BatchPredictResponse
        List of results with total latency.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    total_start = time.perf_counter()

    for text in req.texts:
        inputs = tokenizer(build_prompt(text), return_tensors="pt").to(model.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        action, emotion = generate_constrained(inputs["input_ids"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000

        results.append(
            PredictResponse(
                action=action, emotion=emotion, latency_ms=round(latency, 1)
            )
        )

    total_latency = (time.perf_counter() - total_start) * 1000
    logger.info("predict_batch | count=%d | total=%.0fms", len(req.texts), total_latency)
    return BatchPredictResponse(
        results=results, count=len(results), total_latency_ms=round(total_latency, 1)
    )


@app.get("/actions")
def actions():
    """List all supported robot actions and avatar emotions."""
    return {"actions": ACTIONS, "emotions": EMOTIONS}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "functiongemma-finetuned-g1",
        "device": str(model.device) if model else "not loaded",
    }
