import time
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_extract import extract_constraints
from stage2_filter import filter_products, load_products
import anthropic

DATA_PATH = Path(__file__).parent.parent / "data" / "menu.csv"
FRONTEND_PATH = Path(__file__).parent.parent / "frontend"

RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX    = 10

SEMANTIC_EPSILON = 0.001
MAX_CAFFEINE_MG  = 360

CAFFEINE_RANGES = {
    "none":   (0,   0),
    "low":    (0,   70),
    "medium": (71,  200),
    "high":   (150, float("inf")),
}

app = FastAPI(title="BrewMatch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

products: pd.DataFrame
vectorizer: TfidfVectorizer
product_tfidf_matrix = None
product_ids_list: list
claude_client: anthropic.Anthropic
query_cache: dict = {}
rate_buckets: dict = defaultdict(list)


def _build_product_text(row: pd.Series) -> str:
    dairy_free = "dairy-free no milk" if not row["contains_dairy"] else "contains dairy milk"
    vegan = "vegan plant-based" if row["is_vegan"] else ""
    caff = row["caffeine_mg"]
    if caff == 0:
        caff_desc = "caffeine-free decaf no caffeine"
    elif caff <= 70:
        caff_desc = "low caffeine mild"
    elif caff <= 200:
        caff_desc = "medium caffeine regular"
    else:
        caff_desc = "high caffeine strong bold"
    return (
        f"{row['name']} {row['category']} {row['subcategory']} {row['temperature']} "
        f"{dairy_free} {vegan} {caff_desc} "
        f"{row['caffeine_mg']}mg {row['calories']}cal {row['sugar_g']}g sugar "
        f"${row['price']} {row['description']}"
    )


def _constraint_score(row, constraints: dict) -> float:
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
        scores.append(row["caffeine_mg"] / MAX_CAFFEINE_MG)
    elif caffeine_level in ("low", "none"):
        scores.append(1.0 - row["caffeine_mg"] / MAX_CAFFEINE_MG)
    return sum(scores) / len(scores) if scores else 0.0


def _rank(query: str, filtered_df: pd.DataFrame, constraints: dict) -> list[str]:
    if filtered_df.empty:
        return []

    query_vec = vectorizer.transform([query])
    filtered_ids = filtered_df["product_id"].tolist()
    indices = [product_ids_list.index(pid) for pid in filtered_ids]
    filtered_matrix = product_tfidf_matrix[indices]

    sem_scores = sklearn_cosine(query_vec, filtered_matrix)[0]
    s_min, s_max = sem_scores.min(), sem_scores.max()
    if s_max > s_min:
        sem_scores = (sem_scores - s_min) / (s_max - s_min)
    else:
        sem_scores = np.ones(len(filtered_ids)) * 0.5

    combined = []
    for i, pid in enumerate(filtered_ids):
        row = filtered_df[filtered_df["product_id"] == pid].iloc[0]
        c_score = _constraint_score(row, constraints)
        score = c_score + SEMANTIC_EPSILON * sem_scores[i]
        combined.append((pid, score))

    combined.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in combined]


@app.on_event("startup")
def startup():
    global products, vectorizer, product_tfidf_matrix, product_ids_list, claude_client
    products = load_products(str(DATA_PATH))
    print(f"Loaded {len(products)} drinks")

    texts = [_build_product_text(row) for _, row in products.iterrows()]
    product_ids_list = products["product_id"].tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    product_tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix: {product_tfidf_matrix.shape}")

    claude_client = anthropic.Anthropic()
    print("BrewMatch API ready.")


def check_rate_limit(ip: str):
    now = time.time()
    rate_buckets[ip] = [t for t in rate_buckets[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(rate_buckets[ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Too many requests. Try again in a minute.")
    rate_buckets[ip].append(now)


def cache_key(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


class RankRequest(BaseModel):
    query: str

class RankedDrink(BaseModel):
    rank: int
    product_id: str
    name: str
    category: str
    temperature: str
    caffeine_mg: int
    calories: int
    sugar_g: int
    price: float
    is_vegan: bool
    contains_dairy: bool
    description: str
    constraint_badges: list[str]
    match_score: float

class RankResponse(BaseModel):
    query: str
    constraints: dict
    total_products: int
    filtered_count: int
    ranked: list[RankedDrink]
    elapsed_ms: int
    cached: bool


def _badges(row, c: dict) -> list[str]:
    badges = []
    if c.get("max_calories") and row["calories"] <= c["max_calories"]:
        badges.append(f"{row['calories']} cal ≤ {int(c['max_calories'])} ✓")
    if c.get("max_sugar") and row["sugar_g"] <= c["max_sugar"]:
        badges.append(f"{row['sugar_g']}g sugar ≤ {int(c['max_sugar'])}g ✓")
    if c.get("max_price") and row["price"] <= c["max_price"]:
        badges.append(f"${row['price']} ≤ ${c['max_price']} ✓")
    if c.get("dairy_free") and not row["contains_dairy"]:
        badges.append("dairy-free ✓")
    if c.get("vegan") and row["is_vegan"]:
        badges.append("vegan ✓")
    if c.get("temperature") and row["temperature"] == c["temperature"]:
        badges.append(f"{row['temperature']} ✓")
    return badges


@app.post("/api/rank", response_model=RankResponse)
async def rank(req: RankRequest, request: Request):
    ip = request.client.host
    check_rate_limit(ip)

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(query) > 300:
        raise HTTPException(status_code=400, detail="Query too long (max 300 chars).")

    ck = cache_key(query)
    if ck in query_cache:
        cached = dict(query_cache[ck])
        cached["cached"] = True
        return cached

    t0 = time.time()

    constraints = extract_constraints(query, claude_client)

    filtered = filter_products(constraints, products)
    if filtered.empty:
        for key in ["max_price", "max_sugar", "max_calories"]:
            if constraints.get(key) is not None:
                relaxed = dict(constraints)
                relaxed[key] = None
                filtered = filter_products(relaxed, products)
                if not filtered.empty:
                    break
    if filtered.empty:
        filtered = products

    ranked_ids = _rank(query, filtered, constraints)

    n = len(ranked_ids)
    score_map = {pid: round(1.0 - i / max(n - 1, 1), 3) for i, pid in enumerate(ranked_ids)}

    ranked_drinks = []
    for i, pid in enumerate(ranked_ids[:10]):
        row = products[products["product_id"] == pid].iloc[0]
        ranked_drinks.append(RankedDrink(
            rank=i + 1,
            product_id=pid,
            name=row["name"],
            category=row["category"],
            temperature=row["temperature"],
            caffeine_mg=int(row["caffeine_mg"]),
            calories=int(row["calories"]),
            sugar_g=int(row["sugar_g"]),
            price=float(row["price"]),
            is_vegan=bool(row["is_vegan"]),
            contains_dairy=bool(row["contains_dairy"]),
            description=row["description"],
            constraint_badges=_badges(row.to_dict(), constraints),
            match_score=score_map.get(pid, 0.0),
        ))

    elapsed_ms = int((time.time() - t0) * 1000)
    response = RankResponse(
        query=query,
        constraints=constraints,
        total_products=len(products),
        filtered_count=len(filtered),
        ranked=ranked_drinks,
        elapsed_ms=elapsed_ms,
        cached=False,
    )

    query_cache[ck] = response.model_dump()
    return response


@app.get("/api/examples")
def examples():
    return [
        "strong iced coffee, no dairy please",
        "something sweet and cold but vegan, under $6",
        "hot tea with no caffeine and under 100 cal",
        "a pick-me-up that's not too sweet",
        "blended drink under 400 calories",
        "cold brew, nothing fancy, just black",
        "chai or matcha latte, dairy free",
    ]


@app.get("/api/health")
def health():
    return {"status": "ok", "drinks": len(products)}


app.mount("/", StaticFiles(directory=str(FRONTEND_PATH), html=True), name="frontend")
