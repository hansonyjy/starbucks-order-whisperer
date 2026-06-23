import json
import anthropic

MODEL = "claude-haiku-4-5"

_SYSTEM = """You are a Starbucks order assistant. Extract hard constraints from a customer query.

category values: brewed, cold_brew, espresso, frappuccino, refresher, tea
temperature values: hot, iced, blended
caffeine_level values: none, low, medium, high
  - none: explicitly decaf or no caffeine
  - low: light caffeine, e.g. "not too much caffeine"
  - medium: moderate caffeine
  - high: strong caffeine, "lots of caffeine", "pick me up", "energy"

Set a field to null if the query does not mention that constraint.
temperature must only be set when the customer explicitly requests hot, iced, or blended — not when they describe the weather or other context (e.g. "it's hot out" does NOT set temperature=iced).
dairy_free and vegan are only true when explicitly requested; never false.
max_calories, max_sugar, max_price are numbers (not strings) when present."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "anyOf": [
                {"type": "string", "enum": ["brewed", "cold_brew", "espresso", "frappuccino", "refresher", "tea"]},
                {"type": "null"},
            ]
        },
        "temperature": {
            "anyOf": [
                {"type": "string", "enum": ["hot", "iced", "blended"]},
                {"type": "null"},
            ]
        },
        "max_calories": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "max_sugar":    {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "max_price":    {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "dairy_free":   {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
        "vegan":        {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
        "caffeine_level": {
            "anyOf": [
                {"type": "string", "enum": ["none", "low", "medium", "high"]},
                {"type": "null"},
            ]
        },
    },
    "required": [
        "category", "temperature", "max_calories", "max_sugar",
        "max_price", "dairy_free", "vegan", "caffeine_level",
    ],
    "additionalProperties": False,
}


def extract_constraints(query: str, client: anthropic.Anthropic | None = None) -> dict:
    if client is None:
        client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=_SYSTEM,
        messages=[{"role": "user", "content": query}],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


if __name__ == "__main__":
    import ast
    import pandas as pd

    BOOL_COLS = {"constraint_dairy_free", "constraint_vegan"}

    def _norm_ground_truth(row) -> dict:
        return {
            "category":      row["constraint_category"] if pd.notna(row["constraint_category"]) else None,
            "temperature":   row["constraint_temperature"] if pd.notna(row["constraint_temperature"]) else None,
            "max_calories":  float(row["constraint_max_calories"]) if pd.notna(row["constraint_max_calories"]) else None,
            "max_sugar":     float(row["constraint_max_sugar"]) if pd.notna(row["constraint_max_sugar"]) else None,
            "max_price":     float(row["constraint_max_price"]) if pd.notna(row["constraint_max_price"]) else None,
            "dairy_free":    True if str(row["constraint_dairy_free"]).strip() == "True" else None,
            "vegan":         True if str(row["constraint_vegan"]).strip() == "True" else None,
            "caffeine_level": row["constraint_caffeine_level"] if pd.notna(row["constraint_caffeine_level"]) else None,
        }

    def _field_match(pred, truth) -> bool:
        if truth is None:
            return pred is None
        if pred is None:
            return False
        if isinstance(truth, float):
            return abs(float(pred) - truth) < 1e-6
        return str(pred) == str(truth)

    queries = pd.read_csv("queries_train.csv")
    client = anthropic.Anthropic()

    SAMPLE_IDS = [
        "TRAIN_001", "TRAIN_002", "TRAIN_003", "TRAIN_004", "TRAIN_005",
        "TRAIN_006", "TRAIN_007", "TRAIN_010", "TRAIN_015", "TRAIN_020",
    ]
    sample = queries[queries["query_id"].isin(SAMPLE_IDS)]

    fields = ["category", "temperature", "max_calories", "max_sugar",
              "max_price", "dairy_free", "vegan", "caffeine_level"]
    total_fields = 0
    correct_fields = 0

    print(f"{'ID':<12} {'Match':>5}  Mismatches")
    print("-" * 70)
    for _, row in sample.iterrows():
        gt = _norm_ground_truth(row)
        pred = extract_constraints(row["query_text"], client)

        mismatches = []
        for f in fields:
            total_fields += 1
            if _field_match(pred.get(f), gt[f]):
                correct_fields += 1
            else:
                mismatches.append(f"{f}: got={pred.get(f)!r} want={gt[f]!r}")

        tag = "OK" if not mismatches else "FAIL"
        print(f"{row['query_id']:<12} {tag:>5}  {'; '.join(mismatches) or ''}")

    print("-" * 70)
    print(f"Field accuracy: {correct_fields}/{total_fields} = {correct_fields/total_fields:.1%}")
