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

import json, re, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = './merged-fixed'

ACTIONS = ['stand_up','sit_down','hello','stretch','shake_hand','wave_hand',
           'dance','spin','wiggle_hips','pose','scrape','finger_heart',
           'stand_still','recovery_stand','low_stand','high_stand']

EMOTIONS = ['happy','sad','excited','confused','curious','think']

FUNCTIONS = [
    {'name': 'robot_action', 'description': 'Execute a predefined robot action or gesture', 'parameters': {'type': 'OBJECT', 'properties': {'action_name': {'type': 'STRING', 'enum': ACTIONS}}, 'required': ['action_name']}},
    {'name': 'show_emotion', 'description': 'Display emotion on screen', 'parameters': {'type': 'OBJECT', 'properties': {'emotion': {'type': 'STRING', 'enum': EMOTIONS}}, 'required': ['emotion']}},
]

def build_prompt(user_input):
    decls = ''
    for f in FUNCTIONS:
        decls += (
            f"<start_function_declaration>declaration:{f['name']}"
            f"{{description:<escape>{f['description']}<escape>,"
            f"parameters:{json.dumps(f['parameters'])}}}"
            f"<end_function_declaration>"
        )
    system = 'You are a robot action controller. Always call both robot_action AND show_emotion.\n' + decls
    return f'<bos><start_of_turn>developer\n{system}<end_of_turn>\n<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n'


print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map='auto',
)
model.eval()

# Pre-tokenize all valid action and emotion values
action_token_ids = {}
for a in ACTIONS:
    ids = tokenizer.encode(a, add_special_tokens=False)
    action_token_ids[a] = ids

emotion_token_ids = {}
for e in EMOTIONS:
    ids = tokenizer.encode(e, add_special_tokens=False)
    emotion_token_ids[e] = ids

# Pre-tokenize the template parts
# After prefill, the model should output:
# <start_function_call>call:robot_action{action_name:<escape>ACTION<escape>}<end_function_call>
# <start_function_call>call:show_emotion{emotion:<escape>EMOTION<escape>}<end_function_call><end_of_turn>

# Build prefix for action call
action_prefix = tokenizer.encode(
    '<start_function_call>call:robot_action{action_name:<escape>',
    add_special_tokens=False
)
action_suffix = tokenizer.encode(
    '<escape>}<end_function_call><start_function_call>call:show_emotion{emotion:<escape>',
    add_special_tokens=False
)
emotion_suffix = tokenizer.encode(
    '<escape>}<end_function_call><end_of_turn>',
    add_special_tokens=False
)

print(f'Action prefix tokens: {len(action_prefix)}')
print(f'Action suffix tokens: {len(action_suffix)}')
print(f'Emotion suffix tokens: {len(emotion_suffix)}')
print()

# Show what tokens each action maps to
print('Action token mappings:')
for a, ids in action_token_ids.items():
    print(f'  {a:<20s} -> {ids} ({len(ids)} tokens)')
print()
print('Emotion token mappings:')
for e, ids in emotion_token_ids.items():
    print(f'  {e:<20s} -> {ids} ({len(ids)} tokens)')
print()

# Build constrained token sets (first token of each valid value)
valid_action_first_tokens = set()
for ids in action_token_ids.values():
    valid_action_first_tokens.add(ids[0])

valid_emotion_first_tokens = set()
for ids in emotion_token_ids.values():
    valid_emotion_first_tokens.add(ids[0])


@torch.no_grad()
def generate_constrained(model, input_ids):
    """
    Constrained generation with BATCHED token feeding.

    Key insight: we can feed multiple known tokens in ONE forward pass
    (just like prefill). We only need separate steps at decision points.

    Total: ~3 forward passes instead of 33!
      1. Prefill (input + action_prefix) → pick action
      2. Feed (action_tokens + action_suffix) → pick emotion
      3. Done!
    """
    device = input_ids.device

    # === Forward pass 1: Prefill input + action prefix ===
    prefix_ids = torch.tensor([action_prefix], dtype=torch.long, device=device)
    full_input = torch.cat([input_ids, prefix_ids], dim=1)
    outputs = model(input_ids=full_input, use_cache=True)
    past = outputs.past_key_values

    # Pick action (constrained to valid first tokens)
    logits = outputs.logits[:, -1, :]
    mask = torch.full_like(logits, float('-inf'))
    for tok_id in valid_action_first_tokens:
        mask[0, tok_id] = 0
    action_first_token = (logits + mask).argmax(dim=-1).item()

    # Find which action
    chosen_action = None
    for a, ids in action_token_ids.items():
        if ids[0] == action_first_token:
            chosen_action = a
            break
        
    action_ids = action_token_ids[chosen_action]
    # Combine: action_tokens + action_suffix into one sequence
    combined = action_ids + action_suffix
    combined_tensor = torch.tensor([combined], dtype=torch.long, device=device)
    outputs = model(input_ids=combined_tensor, past_key_values=past, use_cache=True)
    past = outputs.past_key_values

    # Pick emotion (constrained)
    logits = outputs.logits[:, -1, :]
    mask = torch.full_like(logits, float('-inf'))
    for tok_id in valid_emotion_first_tokens:
        mask[0, tok_id] = 0
    emotion_first_token = (logits + mask).argmax(dim=-1).item()

    chosen_emotion = None
    for e, ids in emotion_token_ids.items():
        if ids[0] == emotion_first_token:
            chosen_emotion = e
            break

    return chosen_action, chosen_emotion


# Warmup
print('Warming up...')
for _ in range(3):
    inputs = tokenizer(build_prompt('hello'), return_tensors='pt').to(model.device)
    generate_constrained(model, inputs['input_ids'])
print('Done!\n')

# Benchmark
tests = ['Shake hands!', 'Sit down', 'Dance for me', 'I feel sad', 'Wave hello',
         'Do a spin', 'Good boy!', 'What is that?', 'Stand up', 'You are cute!', 'Wow, you are so cute! Can I shake hand with you?']

print('Running benchmark...\n')
times = []
for t in tests:
    inputs = tokenizer(build_prompt(t), return_tensors='pt').to(model.device)

    torch.cuda.synchronize()
    start = time.time()
    action, emotion = generate_constrained(model, inputs['input_ids'])
    torch.cuda.synchronize()
    ms = (time.time() - start) * 1000

    times.append(ms)
    print(f'{ms:6.0f}ms  {t:<20s} action={action:<16s} emotion={emotion}')

print(f'\n--- Results ---')
print(f'Min:     {min(times):.0f}ms')
print(f'Max:     {max(times):.0f}ms')
print(f'Average: {sum(times)/len(times):.0f}ms')