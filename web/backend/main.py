import time
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_extract import extract_constraints
from stage2_filter import filter_products, load_products
from stage3_rank import precompute_embeddings, rank_products
from sentence_transformers import SentenceTransformer
import anthropic

DATA_PATH = Path(__file__).parent.parent / "data" / "menu.csv"
FRONTEND_PATH = Path(__file__).parent.parent / "frontend"

RATE_LIMIT_WINDOW = 60   # seconds
RATE_LIMIT_MAX    = 10   # requests per window per IP

app = FastAPI(title="BrewMatch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# --- startup: load everything once ---
products: pd.DataFrame
embeddings: dict
model: SentenceTransformer
claude_client: anthropic.Anthropic
query_cache: dict = {}
rate_buckets: dict = defaultdict(list)

@app.on_event("startup")
def startup():
    global products, embeddings, model, claude_client
    products = load_products(str(DATA_PATH))
    print(f"Loaded {len(products)} drinks from {DATA_PATH}")
    print("Loading sentence-transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Pre-computing product embeddings...")
    embeddings = precompute_embeddings(products, model)
    print(f"Embeddings ready for {len(embeddings)} products")
    claude_client = anthropic.Anthropic()
    print("BrewMatch API ready.")


# --- rate limiting ---
def check_rate_limit(ip: str):
    now = time.time()
    bucket = rate_buckets[ip]
    # drop timestamps outside the window
    rate_buckets[ip] = [t for t in bucket if now - t < RATE_LIMIT_WINDOW]
    if len(rate_buckets[ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Too many requests. Try again in a minute.")
    rate_buckets[ip].append(now)


# --- cache key ---
def cache_key(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


# --- request / response models ---
class RankRequest(BaseModel):
    query: str

class ConstraintBadge(BaseModel):
    label: str
    value: str

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
        cached_response = dict(query_cache[ck])
        cached_response["cached"] = True
        return cached_response

    t0 = time.time()

    # Stage 1 — extract constraints via Claude
    constraints = extract_constraints(query, claude_client)

    # Stage 2 — filter
    filtered = filter_products(constraints, products)

    # Progressive fallback
    if filtered.empty:
        fallback_keys = ["max_price", "max_sugar", "max_calories"]
        relaxed = dict(constraints)
        for key in fallback_keys:
            if relaxed.get(key) is not None:
                relaxed[key] = None
                filtered = filter_products(relaxed, products)
                if not filtered.empty:
                    break
    if filtered.empty:
        filtered = products

    # Stage 3 — rank
    ranked_ids = rank_products(query, filtered, embeddings, model, constraints)

    # Build score map (positional, normalised to [0,1])
    n = len(ranked_ids)
    score_map = {pid: round(1.0 - i / max(n - 1, 1), 3) for i, pid in enumerate(ranked_ids)}

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

    ranked_drinks = []
    for i, pid in enumerate(ranked_ids[:10]):
        row = products[products["product_id"] == pid].iloc[0].to_dict()
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
            constraint_badges=_badges(row, constraints),
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

    query_cache[ck] = response.dict()
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


# Serve the frontend for everything else
app.mount("/", StaticFiles(directory=str(FRONTEND_PATH), html=True), name="frontend")
