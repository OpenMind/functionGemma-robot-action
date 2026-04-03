"""
Chat with the FunctionGemma server.
Usage: python3 chat_client.py [--url http://localhost:8200]
"""

import argparse

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8200")
    args = parser.parse_args()

    print(f"Connected to {args.url}")
    print(f"Actions: {requests.get(f'{args.url}/actions').json()}")
    print()

    while True:
        text = input("You: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break

        r = requests.post(f"{args.url}/predict", json={"text": text})
        data = r.json()
        print(
            f"Robot ({data['latency_ms']:.0f}ms): action={data['action']}  emotion={data['emotion']}\n"
        )


if __name__ == "__main__":
    main()
