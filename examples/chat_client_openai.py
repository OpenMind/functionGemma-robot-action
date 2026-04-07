"""
OpenAI-compatible client for FunctionGemma robot-action server.

Demonstrates how to use the /v1/chat/completions endpoint with OpenAI SDK
or standard HTTP requests.
"""

import json
import requests

# Can also use: from openai import OpenAI

SERVER_URL = "http://localhost:8200"


def test_with_requests():
    """Test using standard HTTP requests."""
    print("Testing with requests library...")
    print("-" * 60)

    test_commands = [
        "Hello! Nice to meet you!",
        "Can you wave at me?",
        "Show me you're thinking",
        "I'm confused about something",
    ]

    for command in test_commands:
        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json={
                "model": "functiongemma-finetuned-g1",
                "messages": [{"role": "user", "content": command}],
            },
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n📝 User: {command}")
            print(f"🤖 Model: {data['model']}")

            choice = data["choices"][0]
            if choice.get("message", {}).get("tool_calls"):
                print("🔧 Tool Calls:")
                for tool_call in choice["message"]["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    func_args = json.loads(tool_call["function"]["arguments"])
                    print(f"   • {func_name}({func_args})")

            usage = data.get("usage", {})
            print(
                f"📊 Tokens: {usage.get('prompt_tokens')} prompt + "
                f"{usage.get('completion_tokens')} completion = "
                f"{usage.get('total_tokens')} total"
            )
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")

    print("\n" + "=" * 60)


def test_with_openai_sdk():
    """Test using OpenAI Python SDK (if installed)."""
    try:
        from openai import OpenAI

        print("\nTesting with OpenAI SDK...")
        print("-" * 60)

        # Initialize client pointing to local server
        client = OpenAI(
            base_url=f"{SERVER_URL}/v1",
            api_key="dummy-key",  # Not validated, but required by SDK
        )

        response = client.chat.completions.create(
            model="functiongemma-finetuned-g1",
            messages=[
                {"role": "user", "content": "Wave your hand and show excitement!"}
            ],
        )

        print(f"\n📝 User: Wave your hand and show excitement!")
        print(f"🤖 Model: {response.model}")

        choice = response.choices[0]
        if choice.message.tool_calls:
            print("🔧 Tool Calls:")
            for tool_call in choice.message.tool_calls:
                print(
                    f"   • {tool_call.function.name}"
                    f"({tool_call.function.arguments})"
                )

        print(f"📊 Tokens: {response.usage.prompt_tokens} prompt + "
              f"{response.usage.completion_tokens} completion = "
              f"{response.usage.total_tokens} total")

        print("\n" + "=" * 60)

    except ImportError:
        print(
            "\n⚠️  OpenAI SDK not installed. Install with: pip install openai\n"
        )


def list_models():
    """List available models."""
    print("\nAvailable Models:")
    print("-" * 60)

    response = requests.get(f"{SERVER_URL}/v1/models")
    if response.status_code == 200:
        data = response.json()
        for model in data["data"]:
            print(f"• {model['id']} (owned by {model['owned_by']})")
    else:
        print(f"❌ Error: {response.status_code}")

    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FunctionGemma Robot Action - OpenAI Compatible Client")
    print("=" * 60)

    list_models()
    test_with_requests()
    test_with_openai_sdk()

    print("\n✅ Testing complete!")
