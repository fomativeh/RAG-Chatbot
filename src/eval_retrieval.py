import argparse
import json
from typing import List, Tuple

from dotenv import load_dotenv

from utils import get_or_create_collection


load_dotenv()


def precision_recall_at_k(queries: List[Tuple[str, List[str]]], k: int = 3) -> Tuple[float, float]:
    collection = get_or_create_collection()
    total_prec = 0.0
    total_rec = 0.0
    for query, relevant_sources in queries:
        res = collection.query(query_texts=[query], n_results=k, include=["metadatas"])
        metas = res.get("metadatas", [[]])[0]
        retrieved = [m.get("source") for m in metas]
        retrieved_set = set(retrieved)
        relevant_set = set(relevant_sources)
        tp = len(retrieved_set & relevant_set)
        prec = tp / max(1, len(retrieved))
        rec = tp / max(1, len(relevant_set))
        total_prec += prec
        total_rec += rec
    n = max(1, len(queries))
    return total_prec / n, total_rec / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="eval/retrieval.jsonl")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    queries: List[Tuple[str, List[str]]] = []
    try:
        with open(args.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                queries.append((obj["query"], obj.get("relevant_sources", [])))
    except FileNotFoundError:
        print("No retrieval eval file found; add eval/retrieval.jsonl to use this.")
        return

    prec, rec = precision_recall_at_k(queries, k=args.k)
    print(f"Precision@{args.k}: {prec:.2f}\tRecall@{args.k}: {rec:.2f}")


if __name__ == "__main__":
    main()


