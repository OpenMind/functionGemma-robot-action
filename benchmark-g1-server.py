"""
Benchmark the FunctionGemma server.
Usage: python3 benchmark_client.py [--url http://localhost:8200]
"""

import time
import argparse
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
        requests.post(f"{args.url}/predict", json={"text": "hello"})
    print("Done!\n")

    # Benchmark
    print("Running benchmark...\n")
    times = []
    for t in TESTS:
        start = time.time()
        r = requests.post(f"{args.url}/predict", json={"text": t})
        total_ms = (time.time() - start) * 1000

        data = r.json()
        inference_ms = data["latency_ms"]
        times.append(inference_ms)

        print(f"  {inference_ms:5.0f}ms inference | {total_ms:5.0f}ms total | {t:<30s} -> {data['action']:<16s} {data['emotion']}")

    print(f"\n--- Inference (model only) ---")
    print(f"Min:     {min(times):.0f}ms")
    print(f"Max:     {max(times):.0f}ms")
    print(f"Average: {sum(times)/len(times):.0f}ms")

    # Batch test
    print(f"\n--- Batch test ({len(TESTS)} inputs) ---")
    start = time.time()
    r = requests.post(f"{args.url}/predict_batch", json={"texts": TESTS})
    total_ms = (time.time() - start) * 1000
    data = r.json()
    print(f"Total:   {data['total_latency_ms']:.0f}ms inference | {total_ms:.0f}ms with network")

if __name__ == "__main__":
    main()