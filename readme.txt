# FunctionGemma Robot Actions

A fine-tuned [FunctionGemma 270M](https://huggingface.co/google/functiongemma-270m-it) model that converts natural language commands into structured robot actions and avatar emotions.

## What It Does

```
User: "Can you shake hands with me?"

Robot (44ms): [
  {"function": "robot_action", "args": {"action_name": "shake_hand"}},
  {"function": "show_emotion", "args": {"emotion": "happy"}}
]
```

The model takes a user's voice/text input and outputs:
- **Robot Action** — one of 16 predefined safe actions (shake_hand, dance, spin, sit_down, etc.)
- **Avatar Emotion** — one of 6 emotions (happy, sad, excited, confused, curious, think)

## Supported Actions

| Action | Description |
|--------|-------------|
| stand_up | Stand up |
| sit_down | Sit down |
| hello | Wave hello |
| stretch | Stretch body |
| shake_hand | Handshake |
| wave_hand | Wave hand |
| dance | Dance routine |
| spin | Spin around |
| wiggle_hips | Wag tail / wiggle |
| pose | Strike a pose |
| scrape | Paw the ground |
| finger_heart | Heart sign |
| stand_still | Stay / freeze |
| recovery_stand | Recover from fall |
| low_stand | Crouch low |
| high_stand | Stand tall |

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

| Metric | Value |
|--------|-------|
| Model size | 270M parameters |
| Avg latency (Thor) | ~44ms |


## Setup on NVIDIA Jetson AGX Thor

### 1. Clone this repo and Download the model

```bash
cd ~/Documents/Github
git clone https://github.com/YourOrg/functiongemma-robot-actions.git
cd functiongemma-robot-actions
```
Download the model to local: https://drive.google.com/drive/folders/1a-kwOTcTWzkmFHLGpqpMvH4pSHt0nTKf?usp=sharing

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch transformers accelerate
```

### 3. Test the model with benchmark.python3
python3 benchmark.py
