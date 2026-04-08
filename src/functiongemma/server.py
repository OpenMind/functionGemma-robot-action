"""
FunctionGemma robot-action microservice for NVIDIA Jetson AGX Thor.
Run: uvicorn server:app --host 0.0.0.0 --port 8200.

Uses constrained decoding (2 forward passes instead of 33 autoregressive
steps) for ~59 ms average latency on Thor.

OpenAI-Compatible API
---------------------
POST /v1/chat/completions
    Chat endpoint with function/tool calls in OpenAI format.
GET  /v1/models
    List available models.
GET  /actions
    List supported robot actions and emotions.
GET  /health
    Health check.
"""

import json
import logging
import os
import time
import uuid
from typing import Literal, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("functiongemma-service")

WARMUP_ITERATIONS = int(os.getenv("WARMUP_ITERATIONS", "5"))
MODEL_NAME = os.getenv("MODEL_NAME", "OpenmindAGI/functiongemma-finetuned-g1-multilingual")

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


# OpenAI-Compatible Request/Response Models
class ChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI /v1/chat/completions request format."""

    model: str = "functiongemma-finetuned-g1"
    messages: list[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    max_tokens: Optional[int] = None
    stream: bool = False


class ToolCall(BaseModel):
    """OpenAI tool call format."""

    id: str
    type: Literal["function"] = "function"
    function: dict  # {name: str, arguments: str (JSON)}


class ChatCompletionMessage(BaseModel):
    """OpenAI response message format."""

    role: Literal["assistant"]
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class ChatCompletionChoice(BaseModel):
    """OpenAI choice format."""

    index: int
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "tool_calls"]


class UsageInfo(BaseModel):
    """OpenAI usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI /v1/chat/completions response format."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


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
    outputs = model(input_ids=combined_tensor, past_key_values=past, use_cache=True)

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

    logger.info("=" * 60)
    logger.info("FunctionGemma Robot Actions Server - Starting Up")
    logger.info("=" * 60)

    logger.info(f"Loading FunctionGemma model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    device = next(model.parameters()).device
    logger.info(f"Model loaded on device: {device}")

    # Pre-tokenize action/emotion values
    logger.info("Pre-tokenizing action/emotion templates...")
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

    # Warmup: Run several inference passes to compile CUDA kernels and
    # ensure optimal performance from the first real request
    if WARMUP_ITERATIONS > 0:
        logger.info(f"Warming up model (running {WARMUP_ITERATIONS} inference passes)...")
        warmup_prompts = [
            "hello",
            "wave at me",
            "shake hands",
            "I'm confused",
            "good job!",
        ]

        for i in range(WARMUP_ITERATIONS):
            prompt = warmup_prompts[i % len(warmup_prompts)]
            inputs = tokenizer(build_prompt(prompt), return_tensors="pt").to(model.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            action, emotion = generate_constrained(inputs["input_ids"])

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000

            logger.info(
                f"  Warmup {i+1}/{WARMUP_ITERATIONS}: {elapsed_ms:.0f}ms | {prompt:<15s} -> {action}/{emotion}"
            )
    else:
        logger.info("Warmup disabled (WARMUP_ITERATIONS=0)")

    logger.info("=" * 60)
    logger.info("✓ Server ready! Listening for requests...")
    logger.info("  Endpoints: /v1/chat/completions, /v1/models, /actions, /health")
    logger.info("=" * 60)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Converts OpenAI format requests to FunctionGemma predictions and returns
    results in OpenAI format with tool_calls.

    Parameters
    ----------
    req : ChatCompletionRequest
        OpenAI-formatted request with messages array.

    Returns
    -------
    ChatCompletionResponse
        OpenAI-formatted response with tool calls for action and emotion.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported.")

    # Extract user message (last user message in conversation)
    user_message = None
    for msg in reversed(req.messages):
        if msg.role == "user" and msg.content:
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found.")

    # Run prediction
    inputs = tokenizer(build_prompt(user_message), return_tensors="pt").to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    action, emotion = generate_constrained(inputs["input_ids"])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = (time.perf_counter() - start) * 1000

    # Build OpenAI-format response with tool calls
    tool_calls = [
        ToolCall(
            id=f"call_{uuid.uuid4().hex[:24]}",
            type="function",
            function={
                "name": "robot_action",
                "arguments": json.dumps({"action_name": action}),
            },
        ),
        ToolCall(
            id=f"call_{uuid.uuid4().hex[:24]}",
            type="function",
            function={
                "name": "show_emotion",
                "arguments": json.dumps({"emotion": emotion}),
            },
        ),
    ]

    # Rough token estimate (more accurate would require actual tokenization)
    prompt_tokens = len(inputs["input_ids"][0])
    completion_tokens = 10  # Approximate for function calls

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant", content=None, tool_calls=tool_calls
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    logger.info(
        'chat_completions | text="%s" | action=%s emotion=%s | %.0fms',
        user_message[:50],
        action,
        emotion,
        latency,
    )
    return response


@app.get("/actions")
def actions():
    """List all supported robot actions and avatar emotions."""
    return {"actions": ACTIONS, "emotions": EMOTIONS}


@app.get("/v1/models")
def list_models():
    """OpenAI-compatible models endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": "functiongemma-finetuned-g1",
                "object": "model",
                "created": 1704067200,
                "owned_by": "openmind",
            }
        ],
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": str(model.device) if model else "not loaded",
        "warmup_iterations": WARMUP_ITERATIONS,
    }
