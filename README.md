# ☕ Starbucks Order Whisperer

> **🏆 1st place out of 19 teams — UCLA Starbucks Data Challenge**

A three-stage recommendation engine that turns messy, human, half-caffeinated Starbucks orders like:

> *"yo i need a latte that's max 250 calories and 25g sugar or less"*

…into a ranked list of the drinks you actually want.

This repo is my **solo rebuild** of the winning solution — same idea, cleaner architecture, and a smarter ranking brain. (More on why it's better at the bottom.)

---

## 🧠 The Problem

You're given:
- **115 drinks**, each with category, temperature, caffeine, calories, sugar, price, dietary flags, and a description.
- **Natural-language queries** written the way real people talk — slang, typos, vibes, and all.

Your job: return the products ranked best-match-first. Scored by **NDCG**, so getting the *order* right matters, not just the set.

The catch? People don't speak in SQL. "Something to wake me up that won't break the bank and no dairy please" has to become structured filters *and* a sensible ranking. That's where the pipeline comes in.

---

## 🏗️ How It Works — Three Stages

```
   query: "iced oat-milk latte, keep it under 250 cal"
              │
              ▼
   ┌──────────────────────────┐
   │  1. EXTRACT  (Claude)    │   natural language → structured constraints
   └──────────────────────────┘
              │  { category: espresso, temperature: iced,
              │    dairy_free: true, max_calories: 250 }
              ▼
   ┌──────────────────────────┐
   │  2. FILTER   (pandas)     │   hard rules — only drinks that qualify
   └──────────────────────────┘
              │  8 candidate drinks survive
              ▼
   ┌──────────────────────────┐
   │  3. RANK     (hybrid)     │   constraint margin + semantic tiebreak
   └──────────────────────────┘
              │
              ▼
   ranked list → submission.csv ✅
```

### Stage 1 — Extract 🔍 (`stage1_extract.py`)
Claude (`claude-haiku-4-5`) reads the query and returns constraints as JSON. The trick: it's pinned to a **JSON schema via structured output**, so the model *physically cannot* hand back malformed JSON. No regex, no markdown-fence stripping, no `try/except` prayer circle. There's also a deliberate guard against the classic trap — *"it's hot out"* should **not** set `temperature: iced`.

### Stage 2 — Filter 🧹 (`stage2_filter.py`)
Pure, deterministic pandas. Category, temperature, calories, sugar, price, dairy-free, vegan, and caffeine all become hard filters. Caffeine "levels" map to mg ranges that were **reverse-engineered from the training data** — including a deliberate medium/high overlap at 150–200mg, because that's how the ground truth actually behaves.

### Stage 3 — Rank 🎯 (`stage3_rank.py`)
The interesting part. Ranking is a **hybrid score**:

```
score = constraint_margin  +  ε · semantic_similarity      (ε = 0.001)
```

- **Constraint margin (primary):** how comfortably a drink sits *inside* the limits. Under a 250-cal cap, a 0-cal drink beats a 200-cal one. Caffeine-seeking queries rank by *most* caffeine; "keep it mild" queries rank by *least*.
- **Semantic similarity (tiebreaker):** a local `all-MiniLM-L6-v2` embedding model breaks ties between drinks the constraints can't separate. It's bounded by a tiny ε so it can **never** override a real constraint gap — exactly what the training data says should happen.

No paid embedding API, no rate limits, runs offline.

---

## 🚀 Quickstart

> ⚠️ **Dataset not included.** The product catalog and queries are confidential challenge
> materials and are intentionally left out of this repo. To run the pipeline you'll need
> your own `products.csv`, `queries_train.csv`, and `queries_test.csv` following the schema
> described in [Data Format](#-data-format) below.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key (Stage 1 uses Claude)
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run the full pipeline (point it at your own queries file)
python pipeline.py queries_test.csv
```

Out comes `submission.csv` with a ranked product list for every query.

Each stage is also runnable on its own — every file has a `__main__` block with built-in sanity checks against the training data:

```bash
python stage1_extract.py    # field-level extraction accuracy
python stage2_filter.py     # filter correctness vs. known answers
python stage3_rank.py       # ranking recall on 3 worked examples
```

---

## 📁 Repo Tour

| File | What it does |
|------|--------------|
| `pipeline.py` | Orchestrates all three stages + the fallback logic |
| `stage1_extract.py` | Constraint extraction with Claude + JSON schema |
| `stage2_filter.py` | Deterministic product filtering |
| `stage3_rank.py` | Hybrid constraint-margin + semantic ranking |

*(The challenge dataset — `products.csv`, `queries_train.csv`, `queries_test.csv` — is confidential and not distributed with this repo.)*

---

## 📋 Data Format

The pipeline expects three CSVs. Bring your own following these schemas:

**`products.csv`** — one row per drink:
`product_id, name, category, subcategory, temperature, caffeine_mg, calories, sugar_g, protein_g, contains_dairy, contains_nuts, contains_gluten, is_vegan, description, price`

**`queries_*.csv`** — one row per customer query:
`query_id, query_text` (plus labeled `relevant_products` and `constraint_*` columns in the training set for tuning)

---

## 🛟 Fallback Logic (because real data is mean)

If the hard filters wipe out *every* drink (someone wants a $3 vegan triple-shot under 50 calories), the pipeline doesn't shrug and return nothing. It **progressively relaxes** the most restrictive numeric constraints one at a time — price, then sugar, then calories — until something survives. Last resort: rank the whole menu. You always get an answer.

---

## ✨ Why This Rebuild Beats the Original

The competition version got the win. Then I went back and rebuilt it solo — and made it genuinely better:

- **Bulletproof Stage 1** — JSON-schema structured output replaced fragile prompt-and-parse, killing an entire class of parsing failures.
- **Smarter ranking** — constraint *margin* as the primary signal (grounded in the training labels) instead of leaning on embedding similarity and hand-wavy "simplicity bonuses."
- **Faster & free** — local embeddings replaced a paid, rate-limited embedding API. (The original literally hit `429 Too Many Requests` mid-run.)
- **Actually maintainable** — four focused modules, each with its own tests, instead of one long notebook.

Same problem. Better engineering. ☕

---

*Built for the UCLA × Starbucks Data Challenge. The dataset is confidential and property of the challenge organizers — it is **not** included in this repository. This repo contains only my own pipeline code.*
