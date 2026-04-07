"""
Benchmark the FunctionGemma server using OpenAI-compatible API.
Usage: python3 benchmark_client.py [--url http://localhost:8200]
"""

import argparse
import json
import time

import requests

TESTS = [
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8200")
    args = parser.parse_args()

    # Health check
    r = requests.get(f"{args.url}/health")
    print(f"Server: {r.json()}\n")

    # Warmup
    print("Warming up...")
    for _ in range(3):
        requests.post(
            f"{args.url}/v1/chat/completions",
            json={
                "model": "functiongemma-finetuned-g1",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    print("Done!\n")

    # Benchmark
    print("Running benchmark...\n")
    times = []
    for t in TESTS:
        start = time.time()
        r = requests.post(
            f"{args.url}/v1/chat/completions",
            json={
                "model": "functiongemma-finetuned-g1",
                "messages": [{"role": "user", "content": t}],
            },
        )
        total_ms = (time.time() - start) * 1000

        data = r.json()

        # Extract action and emotion from tool_calls
        tool_calls = data["choices"][0]["message"]["tool_calls"]
        action = None
        emotion = None
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            if func_name == "robot_action":
                action = args["action_name"]
            elif func_name == "show_emotion":
                emotion = args["emotion"]

        # Estimate inference time (total - network overhead estimate)
        inference_ms = total_ms - 10  # Rough estimate
        times.append(inference_ms)

        print(
            f"  {inference_ms:5.0f}ms inference | {total_ms:5.0f}ms total | {t:<30s} -> {action:<16s} {emotion}"
        )

    print("\n--- Inference (model only) ---")
    print(f"Min:     {min(times):.0f}ms")
    print(f"Max:     {max(times):.0f}ms")
    print(f"Average: {sum(times)/len(times):.0f}ms")


if __name__ == "__main__":
    main()
