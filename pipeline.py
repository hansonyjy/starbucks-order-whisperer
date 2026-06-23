import csv
import sys
import anthropic
import pandas as pd
from sentence_transformers import SentenceTransformer

from stage1_extract import extract_constraints
from stage2_filter import load_products, filter_products
from stage3_rank import precompute_embeddings, rank_products

MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_FILE = "submission.csv"


def run_pipeline(
    test_csv: str = "queries_test.csv",
    products_csv: str = "products.csv",
    output_csv: str = OUTPUT_FILE,
) -> None:
    print("Loading products…")
    products = load_products(products_csv)

    print("Loading sentence-transformer model…")
    model = SentenceTransformer(MODEL_NAME, local_files_only=True)

    print("Pre-computing product embeddings…")
    embeddings = precompute_embeddings(products, model)

    queries = pd.read_csv(test_csv)
    client = anthropic.Anthropic()

    results = []
    n = len(queries)
    print(f"Processing {n} queries…\n")

    for i, (_, row) in enumerate(queries.iterrows(), 1):
        qid = row["query_id"]
        text = row["query_text"]
        print(f"[{i:>3}/{n}] {qid}: {text[:70]}")

        constraints = extract_constraints(text, client)
        filtered = filter_products(constraints, products)

        # Fallback: if hard filters eliminate everything, drop the most
        # restrictive numeric constraint one at a time until products remain.
        if filtered.empty:
            fallback_keys = ["max_price", "max_sugar", "max_calories"]
            relaxed = dict(constraints)
            for key in fallback_keys:
                if relaxed.get(key) is not None:
                    relaxed[key] = None
                    filtered = filter_products(relaxed, products)
                    if not filtered.empty:
                        print(f"         ↳ fallback: dropped {key}")
                        break

        # Last resort: use all products
        if filtered.empty:
            filtered = products

        ranked = rank_products(text, filtered, embeddings, model, constraints)
        results.append({"query_id": qid, "products": ";".join(ranked)})
        print(f"         ↳ {len(filtered)} candidates → top: {ranked[:3]}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "products"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Submission saved to {output_csv} ({len(results)} rows)")


if __name__ == "__main__":
    test_csv = sys.argv[1] if len(sys.argv) > 1 else "queries_test.csv"
    run_pipeline(test_csv=test_csv)
