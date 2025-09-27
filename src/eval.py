import argparse
import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from utils import get_models, get_openrouter_client


load_dotenv()


def heuristic_is_supported(client: OpenAI, model: str, question: str, context: str, answer: str) -> bool:
    prompt = (
        "Decide if the answer is supported by the context.\n\n"
        f"Question: {question}\n\nContext: {context}\n\nAnswer: {answer}\n\n"
        "Respond with one word: YES if fully supported, otherwise NO."
    )
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0)
    text = (resp.choices[0].message.content or "").strip().upper()
    return text.startswith("Y")


def run_eval(path: str, limit: int) -> None:
    embedding_model, chat_model = get_models()
    client = get_openrouter_client()

    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))

    items = items[:limit] if limit else items
    if not items:
        print("No eval items found.")
        return

    num_yes = 0
    for i, ex in enumerate(items, 1):
        ok = heuristic_is_supported(client, chat_model, ex["question"], ex["context"], ex["answer"])
        num_yes += 1 if ok else 0
        print(f"[{i}/{len(items)}] {'OK' if ok else 'FAIL'} - {ex['question'][:60]}...")

    print(f"Supported: {num_yes}/{len(items)} ({100.0 * num_yes/len(items):.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="eval/qa.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    run_eval(args.path, args.limit)


if __name__ == "__main__":
    main()


