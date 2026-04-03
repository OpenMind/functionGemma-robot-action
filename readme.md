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

## Files

| File | Description |
|------|-------------|
| `train.py` | Fine-tuning script (LoRA on FunctionGemma 270M) |
| `remerge.py` | Re-merge LoRA adapter with vocab size fix |
| `chat.py` | Interactive chat using standard generation |
| `benchmark.py` | Constrained decoding benchmark |
| `train-g1.jsonl` | Training data (545 examples) |

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

## Setup on NVIDIA Jetson AGX Thor

### 1. Clone this repo

```bash
cd ~/Documents/Github
git clone https://github.com/YourOrg/functiongemma-robot-actions.git
cd functiongemma-robot-actions
```

### 2. Download the model

Download the functionGemma-finetuned-g1 model to the repo directory:
1. Google Drive: https://drive.google.com/drive/folders/1Jx5zfi_Hixq6ABJCBF5yQm3nox9CVEeA?usp=sharing

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

### 3. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch transformers accelerate
```

### 4. Run benchmark

```bash
python3 benchmark-g1.py
python3 benchmark-g1-server.py # if docker env already launched the server
```

### 5. Interactive chat

```bash
python3 chat-g1.py
python3 chat-client.py # if docker env already launched the server
```
