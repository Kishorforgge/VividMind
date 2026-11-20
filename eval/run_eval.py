"""Compare vector-only vs hybrid retrieval on your own test questions.

Run from the project folder:   python eval/run_eval.py

For every answerable question we check whether the correct page of the correct
file appears in the top-k retrieved passages ("hit rate @k"). We also compute
MRR (how high the correct page ranks). For unanswerable questions we measure how
often VividMind would refuse to answer at the chosen strictness.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retriever import get_collection, load_embedder, retrieve  # noqa: E402

K_VALUES = (1, 3, 5)
STRICTNESS = 0.1  # keep equal to the default slider value in app.py


def is_correct(h, q):
    """A retrieved passage is correct if it holds the answer.

    Two ways to define the answer in questions.json:
      * "contains": ["word or phrase", ...]  -> passage text includes any of them
        (works for a single-page PDF)
      * "file" + "pages": [..]               -> passage comes from that file/page
    """
    if q.get("contains"):
        text = h["text"].lower()
        return any(s.lower() in text for s in q["contains"])
    return h["file"] == q["file"] and h["page"] in q["pages"]


def first_correct_rank(hits, q):
    """1-based rank of the first correct passage, else None."""
    for rank, h in enumerate(hits, start=1):
        if is_correct(h, q):
            return rank
    return None


def evaluate(questions, collection, embedder, hybrid):
    answerable = [q for q in questions if not q.get("unanswerable")]
    unanswerable = [q for q in questions if q.get("unanswerable")]

    ranks = []
    for q in answerable:
        hits = retrieve(q["question"], collection, embedder, k=max(K_VALUES), hybrid=hybrid)
        ranks.append(first_correct_rank(hits, q))

    result = {"n_answerable": len(answerable)}
    for k in K_VALUES:
        hit = sum(1 for r in ranks if r is not None and r <= k)
        result[f"hit@{k}"] = hit / len(answerable) if answerable else 0.0
    result["mrr"] = (
        sum(1 / r for r in ranks if r is not None) / len(answerable) if answerable else 0.0
    )

    refused = 0
    for q in unanswerable:
        hits = retrieve(q["question"], collection, embedder, k=max(K_VALUES), hybrid=hybrid)
        best = max((h["similarity"] for h in hits), default=0.0)
        if best < STRICTNESS:
            refused += 1
    result["n_unanswerable"] = len(unanswerable)
    result["refusal_rate"] = refused / len(unanswerable) if unanswerable else None

    # which questions failed (useful for error analysis)
    result["misses"] = [
        q["question"] for q, r in zip(answerable, ranks) if r is None or r > max(K_VALUES)
    ]
    return result


def main():
    path = Path(__file__).with_name("questions.json")
    questions = json.loads(path.read_text(encoding="utf-8"))

    collection = get_collection()
    if collection.count() == 0:
        sys.exit("The database is empty. Process your PDFs in the app first.")
    embedder = load_embedder()

    vec = evaluate(questions, collection, embedder, hybrid=False)
    hyb = evaluate(questions, collection, embedder, hybrid=True)

    print(f"\nAnswerable questions (N): {vec['n_answerable']}")
    print(f"{'Metric':<16}{'Vector only':>14}{'Hybrid':>14}")
    print("-" * 44)
    for k in K_VALUES:
        print(f"{'Hit rate @' + str(k):<16}{vec[f'hit@{k}']:>13.0%}{hyb[f'hit@{k}']:>14.0%}")
    print(f"{'MRR':<16}{vec['mrr']:>14.3f}{hyb['mrr']:>14.3f}")

    if vec["n_unanswerable"]:
        print(
            f"\nUnanswerable questions: {vec['n_unanswerable']} | "
            f"correctly refused at strictness {STRICTNESS}: "
            f"vector {vec['refusal_rate']:.0%}, hybrid {hyb['refusal_rate']:.0%}"
        )

    print("\nQuestions the hybrid retriever missed:")
    for m in hyb["misses"] or ["(none)"]:
        print(" -", m)

    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps({"vector": vec, "hybrid": hyb}, indent=2), encoding="utf-8")
    print(f"\nSaved details to {out.name}")


if __name__ == "__main__":
    main()