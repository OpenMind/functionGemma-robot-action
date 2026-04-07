"""
Benchmark the FunctionGemma server (multilingual) using OpenAI-compatible API.
Usage: python3 benchmark-g1-server.py [--url http://localhost:8200]
"""

import argparse
import json
import time

import requests

TESTS = [
    # English
    {"text": "Shake hands!", "action": "shake_hand", "lang": "en"},
    {"text": "Wave at me", "action": "face_wave", "lang": "en"},
    {"text": "Put your hands up", "action": "hands_up", "lang": "en"},
    {"text": "I feel sad", "action": "stand_still", "lang": "en"},
    {"text": "Show me your hand", "action": "show_hand", "lang": "en"},
    {"text": "Just stand there", "action": "stand_still", "lang": "en"},
    {"text": "What is that?", "action": "stand_still", "lang": "en"},
    {"text": "You are cute!", "action": "stand_still", "lang": "en"},
    {"text": "Nice to meet you", "action": "shake_hand", "lang": "en"},
    {"text": "Hold up the Visa card for payment", "action": "show_hand", "lang": "en"},
    # Chinese
    {"text": "跟我握手", "action": "shake_hand", "lang": "zh"},
    {"text": "你好！", "action": "face_wave", "lang": "zh"},
    {"text": "把双手举起来", "action": "hands_up", "lang": "zh"},
    {"text": "我今天心情不好", "action": "stand_still", "lang": "zh"},
    {"text": "给我看你的手", "action": "show_hand", "lang": "zh"},
    {"text": "你叫什么名字？", "action": "stand_still", "lang": "zh"},
    {"text": "再见！", "action": "face_wave", "lang": "zh"},
    {"text": "出示你的Visa卡", "action": "show_hand", "lang": "zh"},
    {"text": "我升职了！", "action": "stand_still", "lang": "zh"},
    {"text": "这是什么意思？", "action": "stand_still", "lang": "zh"},
    # Japanese
    {"text": "握手してください", "action": "shake_hand", "lang": "ja"},
    {"text": "こんにちは！", "action": "face_wave", "lang": "ja"},
    {"text": "手を上げて", "action": "hands_up", "lang": "ja"},
    {"text": "今日は気分が悪い", "action": "stand_still", "lang": "ja"},
    {"text": "手を見せて", "action": "show_hand", "lang": "ja"},
    {"text": "名前は何ですか？", "action": "stand_still", "lang": "ja"},
    {"text": "さようなら", "action": "face_wave", "lang": "ja"},
    {"text": "カードで支払って", "action": "show_hand", "lang": "ja"},
    {"text": "疲れた", "action": "stand_still", "lang": "ja"},
    {"text": "あれは何ですか？", "action": "stand_still", "lang": "ja"},
    # French
    {"text": "Serrez-moi la main", "action": "shake_hand", "lang": "fr"},
    {"text": "Bonjour !", "action": "face_wave", "lang": "fr"},
    {"text": "Lève les mains", "action": "hands_up", "lang": "fr"},
    {"text": "Je suis triste", "action": "stand_still", "lang": "fr"},
    {"text": "Montre-moi ta main", "action": "show_hand", "lang": "fr"},
    {"text": "Comment tu t'appelles ?", "action": "stand_still", "lang": "fr"},
    {"text": "Au revoir !", "action": "face_wave", "lang": "fr"},
    {"text": "Montre la carte Visa", "action": "show_hand", "lang": "fr"},
    {"text": "Je suis fatigué", "action": "stand_still", "lang": "fr"},
    {"text": "Raconte-moi une blague", "action": "stand_still", "lang": "fr"},
    # German
    {"text": "Gib mir die Hand", "action": "shake_hand", "lang": "de"},
    {"text": "Hallo!", "action": "face_wave", "lang": "de"},
    {"text": "Hände hoch!", "action": "hands_up", "lang": "de"},
    {"text": "Ich bin traurig", "action": "stand_still", "lang": "de"},
    {"text": "Zeig mir deine Hand", "action": "show_hand", "lang": "de"},
    {"text": "Wie heißt du?", "action": "stand_still", "lang": "de"},
    {"text": "Tschüss!", "action": "face_wave", "lang": "de"},
    {"text": "Zeig die Visa-Karte", "action": "show_hand", "lang": "de"},
    {"text": "Ich bin müde", "action": "stand_still", "lang": "de"},
    {"text": "Erzähl mir einen Witz", "action": "stand_still", "lang": "de"},
    # Spanish
    {"text": "Dame la mano", "action": "shake_hand", "lang": "es"},
    {"text": "¡Hola!", "action": "face_wave", "lang": "es"},
    {"text": "¡Manos arriba!", "action": "hands_up", "lang": "es"},
    {"text": "Estoy triste", "action": "stand_still", "lang": "es"},
    {"text": "Muéstrame tu mano", "action": "show_hand", "lang": "es"},
    {"text": "¿Cómo te llamas?", "action": "stand_still", "lang": "es"},
    {"text": "¡Adiós!", "action": "face_wave", "lang": "es"},
    {"text": "Muestra la tarjeta Visa", "action": "show_hand", "lang": "es"},
    {"text": "Estoy cansado", "action": "stand_still", "lang": "es"},
    {"text": "Cuéntame un chiste", "action": "stand_still", "lang": "es"},
]

LANG_NAMES = {
    "en": "English", "zh": "Chinese", "ja": "Japanese",
    "fr": "French", "de": "German", "es": "Spanish",
}


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
    print("Running multilingual benchmark...\n")
    lang_stats = {}

    for t in TESTS:
        start = time.time()
        r = requests.post(
            f"{args.url}/v1/chat/completions",
            json={
                "model": "functiongemma-finetuned-g1",
                "messages": [{"role": "user", "content": t["text"]}],
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

        inference_ms = total_ms - 10  # Rough estimate
        lang = t["lang"]
        expected = t["action"]
        correct = action == expected
        status = "✓" if correct else "✗"

        if lang not in lang_stats:
            lang_stats[lang] = {"correct": 0, "total": 0, "inference": [], "network": []}
        lang_stats[lang]["total"] += 1
        lang_stats[lang]["inference"].append(inference_ms)
        lang_stats[lang]["network"].append(total_ms)
        if correct:
            lang_stats[lang]["correct"] += 1

        print(
            f"  {lang} {status} {inference_ms:5.0f}ms inf | {total_ms:5.0f}ms total | "
            f"{t['text']:<35s} expect={expected:<16s} got={action:<16s} {emotion}"
        )

    # Latency
    all_inf = [ms for s in lang_stats.values() for ms in s["inference"]]
    all_net = [ms for s in lang_stats.values() for ms in s["network"]]

    print(f"\n{'=' * 70}")
    print("LATENCY")
    print(f"{'=' * 70}")
    print(f"  Inference:  min={min(all_inf):.0f}ms  max={max(all_inf):.0f}ms  avg={sum(all_inf)/len(all_inf):.0f}ms")
    print(f"  Total:      min={min(all_net):.0f}ms  max={max(all_net):.0f}ms  avg={sum(all_net)/len(all_net):.0f}ms")
    print()
    for lang, stats in sorted(lang_stats.items()):
        inf_avg = sum(stats["inference"]) / len(stats["inference"])
        net_avg = sum(stats["network"]) / len(stats["network"])
        name = LANG_NAMES.get(lang, lang)
        print(f"  {name:<10s} inference={inf_avg:.0f}ms  total={net_avg:.0f}ms")

    # Accuracy
    total_correct = sum(s["correct"] for s in lang_stats.values())
    total = sum(s["total"] for s in lang_stats.values())

    print(f"\n{'=' * 70}")
    print("ACCURACY")
    print(f"{'=' * 70}")
    print(f"  Overall: {total_correct}/{total} ({total_correct/total*100:.0f}%)\n")

    for lang, stats in sorted(lang_stats.items()):
        pct = stats["correct"] / stats["total"] * 100
        name = LANG_NAMES.get(lang, lang)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {name:<10s} {bar} {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    print()  # Final newline


if __name__ == "__main__":
    main()
