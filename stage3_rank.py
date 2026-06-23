import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

MAX_CAFFEINE_MG = 360  # dataset maximum (Blonde Roast)

# Semantic is a pure tiebreaker — it must never override a real constraint score gap.
# Analysis of training ground truth shows primary rank = avg normalised constraint margin;
# within tied groups the ordering matches semantic similarity (TRAIN_005 disproves caffeine
# as the tiebreaker: CBR_002 has the highest caffeine but ranks last in its tie group).
SEMANTIC_EPSILON = 0.001  # max semantic contribution = 0.001 < smallest real constraint gap


def build_product_text(row: pd.Series) -> str:
    dairy_free = "dairy-free" if not row["contains_dairy"] else "contains dairy"
    vegan = "vegan" if row["is_vegan"] else "not vegan"
    return (
        f"{row['name']}. "
        f"{row['category']} {row['subcategory']}, {row['temperature']}. "
        f"{dairy_free}, {vegan}. "
        f"{row['caffeine_mg']}mg caffeine, {row['calories']} calories, "
        f"{row['sugar_g']}g sugar, ${row['price']}. "
        f"{row['description']}"
    )


def precompute_embeddings(
    products_df: pd.DataFrame,
    model: SentenceTransformer,
) -> dict[str, np.ndarray]:
    """Embed all products once; returns {product_id: unit-norm embedding}."""
    texts = [build_product_text(row) for _, row in products_df.iterrows()]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return dict(zip(products_df["product_id"], embeddings))


def _constraint_score(row: pd.Series, constraints: dict) -> float:
    """
    Score in [0, 1] reflecting how comfortably a product fits within the constraints.
    Higher = product sits further from the constraint limits (i.e. 'more room to spare').
    Products with 0 calories beat products with 200 calories when max_calories=250.
    """
    scores = []

    if constraints.get("max_calories") is not None:
        limit = constraints["max_calories"]
        scores.append((limit - row["calories"]) / limit)

    if constraints.get("max_sugar") is not None:
        limit = constraints["max_sugar"]
        scores.append((limit - row["sugar_g"]) / limit)

    if constraints.get("max_price") is not None:
        limit = constraints["max_price"]
        scores.append((limit - row["price"]) / limit)

    caffeine_level = constraints.get("caffeine_level")
    if caffeine_level in ("high", "medium"):
        # Training data: high-caffeine queries rank by most caffeine first
        scores.append(row["caffeine_mg"] / MAX_CAFFEINE_MG)
    elif caffeine_level in ("low", "none"):
        # Low-caffeine queries rank by least caffeine first
        scores.append(1.0 - row["caffeine_mg"] / MAX_CAFFEINE_MG)

    # Return 0.0 (not 0.5) when no numeric constraints are active so that
    # SEMANTIC_EPSILON * semantic_score becomes the sole ranking signal,
    # which is what TRAIN_006 (dairy_free only, no numeric constraints) shows.
    return sum(scores) / len(scores) if scores else 0.0


def rank_products(
    query: str,
    filtered_df: pd.DataFrame,
    product_embeddings: dict[str, np.ndarray],
    model: SentenceTransformer,
    constraints: dict | None = None,
) -> list[str]:
    """Return filtered product IDs sorted best-match-first using a hybrid score."""
    if filtered_df.empty:
        return []

    constraints = constraints or {}
    query_emb = model.encode(query, normalize_embeddings=True)

    ids = filtered_df["product_id"].tolist()
    candidate_matrix = np.stack([product_embeddings[pid] for pid in ids])
    semantic_scores = candidate_matrix @ query_emb  # cosine sim, already in [-1,1]

    # Rescale semantic scores to [0, 1]
    s_min, s_max = semantic_scores.min(), semantic_scores.max()
    if s_max > s_min:
        semantic_scores = (semantic_scores - s_min) / (s_max - s_min)
    else:
        semantic_scores = np.ones(len(ids)) * 0.5

    combined = []
    for i, (pid, row) in enumerate(zip(ids, filtered_df.itertuples(index=False))):
        c_score = _constraint_score(row._asdict(), constraints)
        # Constraint margin is primary; semantic is a tiny tiebreaker.
        # With SEMANTIC_EPSILON=0.001 and semantic in [0,1], semantic can never
        # flip two products whose constraint scores differ by more than 0.001.
        score = c_score + SEMANTIC_EPSILON * semantic_scores[i]
        combined.append((pid, score))

    combined.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in combined]


if __name__ == "__main__":
    from stage2_filter import load_products, filter_products

    print("Loading model…")
    model = SentenceTransformer(MODEL_NAME, local_files_only=True)

    products = load_products("products.csv")

    print("Pre-computing product embeddings…")
    embeddings = precompute_embeddings(products, model)
    print(f"  {len(embeddings)} products embedded\n")

    # --- Test 1: TRAIN_021 (numeric constraints dominate) ---
    query = "yo i need a latte that's max 250 calories and 25g sugar or less"
    constraints = {"category": "espresso", "max_calories": 250, "max_sugar": 25}
    expected = [
        "ESP_014", "ESP_002", "ICE_001", "ESP_001", "ESP_015",
        "ICE_018", "ESP_016", "ICE_015", "ICE_017", "ESP_017",
        "ICE_012", "ICE_009", "ICE_013", "ICE_002", "ICE_016",
        "ESP_008", "ICE_007", "ICE_008", "ESP_009", "ESP_018",
        "ESP_003", "ESP_010",
    ]
    filtered = filter_products(constraints, products)
    ranked = rank_products(query, filtered, embeddings, model, constraints)
    top5_overlap = len(set(ranked[:5]) & set(expected[:5]))
    print("TRAIN_021 (latte ≤250cal, ≤25g sugar)")
    print(f"  Ranked   : {ranked}")
    print(f"  Expected : {expected}")
    print(f"  Recall   : {len(set(ranked) & set(expected))}/{len(expected)}")
    print(f"  Top-5 overlap: {top5_overlap}/5\n")

    # --- Test 2: TRAIN_001 (caffeine constraint dominates) ---
    query2 = "need a pick me up, something like just black coffee that's need the caffeine and under 45 grams of sugar"
    constraints2 = {"category": "brewed", "max_sugar": 45, "caffeine_level": "high"}
    expected2 = ["BRW_002", "BRW_001", "BRW_005", "BRW_003"]
    filtered2 = filter_products(constraints2, products)
    ranked2 = rank_products(query2, filtered2, embeddings, model, constraints2)
    top5_2 = len(set(ranked2[:4]) & set(expected2[:4]))
    print("TRAIN_001 (brewed, high caffeine, ≤45g sugar)")
    print(f"  Ranked   : {ranked2}")
    print(f"  Expected : {expected2}")
    print(f"  Top-4 overlap: {top5_2}/4\n")

    # --- Test 3: TRAIN_005 (vegan cold brew under 100 cal) ---
    query3 = "could i get a cold brew that's under 100 cal and vegan friendly please?"
    constraints3 = {"category": "cold_brew", "max_calories": 100, "vegan": True}
    expected3 = ["CBR_001", "CBR_012", "CBR_002", "CBR_011", "CBR_009"]
    filtered3 = filter_products(constraints3, products)
    ranked3 = rank_products(query3, filtered3, embeddings, model, constraints3)
    top5_3 = len(set(ranked3[:5]) & set(expected3[:5]))
    print("TRAIN_005 (cold brew, vegan, ≤100cal)")
    print(f"  Ranked   : {ranked3}")
    print(f"  Expected : {expected3}")
    print(f"  Top-5 overlap: {top5_3}/5")
