# FunctionGemma Robot Actions

A fine-tuned [FunctionGemma 270M](https://huggingface.co/google/functiongemma-270m-it) model that converts natural language commands into structured robot actions and avatar emotions.

## What It Does

```
User: "Can you shake hands with me?"

Robot (56ms): [
  {"function": "robot_action", "args": {"action_name": "shake_hand"}},
  {"function": "show_emotion", "args": {"emotion": "happy"}}
]
```

The model takes a user's voice/text input and outputs:
- **Robot Action** — one of 5 predefined safe actions
- **Avatar Emotion** — one of 6 Rive-animated emotions displayed on the robot's screen

For general questions or conversation, the robot defaults to `stand_still` with an appropriate emotion.

## Quick Start

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install dependencies
git clone https://github.com/OpenMind/functionGemma-robot-action.git
cd functionGemma-robot-action
uv sync

# 3. Run the server
uv run uvicorn functiongemma.server:app --host 0.0.0.0 --port 8200
# or: make server

# 4. Test it!
uv run python examples/chat_client_openai.py
# or: make example
```

## OpenAI-Compatible API

The server provides OpenAI-compatible endpoints, allowing you to use the OpenAI SDK or any OpenAI-compatible client:

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat endpoint with tool calls in OpenAI format |
| `/v1/models` | GET | List available models |
| `/actions` | GET | List supported actions and emotions |
| `/health` | GET | Health check |

### Example with OpenAI SDK

```python
from openai import OpenAI

# Point to your local server
client = OpenAI(
    base_url="http://localhost:8200/v1",
    api_key="dummy-key"  # Not validated
)

response = client.chat.completions.create(
    model="functiongemma-finetuned-g1",
    messages=[
        {"role": "user", "content": "Wave at me!"}
    ]
)

# Response includes tool_calls
for tool_call in response.choices[0].message.tool_calls:
    print(f"{tool_call.function.name}: {tool_call.function.arguments}")
# Output:
# robot_action: {"action_name": "face_wave"}
# show_emotion: {"emotion": "happy"}
```

### Example with HTTP Requests

```bash
curl -X POST http://localhost:8200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "functiongemma-finetuned-g1",
    "messages": [
      {"role": "user", "content": "Hello! Nice to meet you!"}
    ]
  }'
```

### Testing

Run the OpenAI-compatible test client:

```bash
python3 chat_client_openai.py
```

## Supported Actions

| Action | Description |
|--------|-------------|
| shake_hand | Handshake gesture |
| face_wave | Wave hello |
| hands_up | Raise both hands |
| stand_still | Stay idle (default for general conversation) |
| show_hand | Show open hand |

## Supported Emotions

| Emotion | Rive Animation |
|---------|---------------|
| happy | Happy.riv |
| sad | Sad.riv |
| excited | Excited.riv |
| confused | Confused.riv |
| curious | Curious.riv |
| think | Think.riv |

## Performance

Benchmarked on NVIDIA Jetson AGX Thor with constrained decoding (`benchmark.py`):

| Metric | Value |
|--------|-------|
| Model size | 270M parameters |
| Min latency | ~52ms |
| Max latency | ~72ms |
| Avg latency | ~59ms |

The constrained decoding approach reduces autoregressive generation from ~33 tokens down to 2 forward passes (one for action, one for emotion), achieving ~18x speedup over standard `model.generate()`.

## Project Structure

```
functiongemma-robot-action/
├── src/
│   └── functiongemma/
│       ├── __init__.py
│       └── server.py          # FastAPI server with OpenAI-compatible API
├── scripts/
│   ├── train-g1.py            # Fine-tuning script (LoRA on FunctionGemma 270M)
│   └── chat-g1.py             # Interactive chat using standard generation
├── benchmarks/
│   ├── benchmark-g1.py        # Constrained decoding benchmark (local)
│   ├── benchmark-g1-server.py # Benchmark against server API
│   └── benchmark-g1-server-multilingual.py  # Multilingual benchmark
├── examples/
│   └── chat_client_openai.py  # OpenAI-compatible client for testing
├── data/
│   └── train-g1.jsonl         # Training data (545 examples)
├── docker/
│   └── Dockerfile.functiongemma
├── docker-compose.yml
└── readme.md
```

## Training

Trained with LoRA on an NVIDIA RTX 5070 Ti (16 GB):

| Parameter | Value |
|-----------|-------|
| Base model | google/functiongemma-270m-it |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Epochs | 5 |
| Learning rate | 2e-4 |
| Batch size | 2 (effective 4 with grad accum) |
| Training examples | 545 (490 train / 55 eval) |
| Max sequence length | 512 |

To train your own model:

```bash
# Install training dependencies
uv sync --extra training
# or: make install-training

# Run training
uv run python scripts/train-g1.py
# or: make train
```

The training script will fine-tune the base FunctionGemma model and save the result to `./functiongemma-robot-actions/`.

## Setup on NVIDIA Jetson AGX Thor

### 1. Clone this repo

```bash
cd ~/Documents/Github
git clone https://github.com/YourOrg/functiongemma-robot-actions.git
cd functiongemma-robot-actions
```

### 2. Download the model

Download the functionGemma-finetuned-g1 model to the repo directory:
1. Huggingface: OpenmindAGI/functiongemma-finetuned-g1 (English Primary Model)
2. Huggingface: OpenmindAGI/functiongemma-finetuned-g1-multilingual (supports English, Japanese, Chinese, French, German, Spanish)

Place it so the directory structure looks like:

```
functiongemma-robot-actions/
├── functionGemma-finetuned-g1/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
├── benchmark.py
├── chat.py
└── ...
```

### 3. Install dependencies

#### Option A: Using uv (recommended - fast!)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install with optional dependencies
uv sync --all-extras  # Includes training, benchmarks, and dev tools
```

See [UV_GUIDE.md](UV_GUIDE.md) for more uv commands and usage.

#### Option B: Using pip

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch transformers accelerate fastapi uvicorn
```

### 4. Run the server

#### Using Make (easiest)
```bash
make server  # Runs the server with auto-reload
```

#### Using uv
```bash
uv run uvicorn functiongemma.server:app --host 0.0.0.0 --port 8200
```

#### Using Docker
```bash
docker-compose up
# or
make docker-run
```

### 5. Run benchmarks and examples

```bash
# Using Make
make benchmark          # Server benchmark
make benchmark-local    # Local model benchmark
make example           # OpenAI client example

# Using uv directly
uv run python benchmarks/benchmark-g1-server.py
uv run python benchmarks/benchmark-g1-server-multilingual.py
uv run python examples/chat_client_openai.py

# Interactive chat
uv run python scripts/chat-g1.py
# or
make chat
```
