import pandas as pd

# Caffeine thresholds derived from training data patterns:
# "none"=0mg, "low"<=70mg, "medium"=71-200mg, "high">=150mg
# Medium/high intentionally overlap at 150-200mg (confirmed by training examples
# where 150mg espresso shots appear in both medium and high results)
CAFFEINE_RANGES = {
    "none":   (0,   0),
    "low":    (0,   70),
    "medium": (71,  200),
    "high":   (150, float("inf")),
}

# "coffee" is an umbrella category spanning every coffee preparation.
COFFEE_CATEGORIES = {"brewed", "cold_brew", "espresso"}

# Subcategories that are inherently milk-based (dairy or plant milk).
_MILK_SUBCATEGORIES = {
    "latte", "mocha", "macchiato", "flat_white", "cappuccino",
    "cortado", "creme", "steamer", "hot_chocolate", "coconut",
}
_PLANT_MILK_RE   = r"oatmilk|almondmilk|coconutmilk|oat milk|almond milk|coconut milk|soy"
_MILK_TOPPING_RE = r"cold foam|sweet cream|whipped cream"


def _compute_contains_milk(df: pd.DataFrame) -> pd.Series:
    """True when a drink contains milk of any kind (dairy OR plant).

    contains_dairy only flags dairy, so plant-milk drinks (oatmilk/almond/coconut
    lattes, coconutmilk refreshers) and milk-based subcategories must be detected
    separately. Used for "black"/"no milk" requests, where any milk should be
    excluded. Matching is ingredient-based (subcategory, plant-milk in name, milk
    toppings) rather than scanning the description for the word "milk", which
    misfires on prose like "perfect with or without milk".
    """
    name = df["name"].fillna("").str.lower()
    desc = df["description"].fillna("").str.lower()
    return (
        df["contains_dairy"]
        | df["subcategory"].isin(_MILK_SUBCATEGORIES)
        | name.str.contains(_PLANT_MILK_RE, regex=True)
        | (name + " " + desc).str.contains(_MILK_TOPPING_RE, regex=True)
    )


def load_products(path: str = "products.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize boolean columns that pandas may read as strings
    for col in ("contains_dairy", "contains_nuts", "contains_gluten", "is_vegan"):
        df[col] = df[col].map(lambda v: v if isinstance(v, bool) else str(v).strip() == "True")
    df["contains_milk"] = _compute_contains_milk(df)
    return df


def filter_products(constraints: dict, products_df: pd.DataFrame) -> pd.DataFrame:
    df = products_df.copy()

    category = constraints.get("category")
    if category is not None:
        if category == "coffee":
            # "coffee" is an umbrella across all coffee preparations. This matters
            # for e.g. "iced coffee": brewed has no iced options, so the match must
            # be able to span cold_brew (cold brew/nitro) and espresso (iced
            # americano, iced shaken espresso) too — not just one of them.
            df = df[df["category"].isin(COFFEE_CATEGORIES)]
        else:
            df = df[df["category"] == category]

    temperature = constraints.get("temperature")
    if temperature is not None:
        df = df[df["temperature"] == temperature]

    max_calories = constraints.get("max_calories")
    if max_calories is not None:
        df = df[df["calories"] <= max_calories]

    max_sugar = constraints.get("max_sugar")
    if max_sugar is not None:
        df = df[df["sugar_g"] <= max_sugar]

    max_price = constraints.get("max_price")
    if max_price is not None:
        df = df[df["price"] <= max_price]

    if constraints.get("dairy_free") is True:
        df = df[df["contains_dairy"] == False]

    if constraints.get("vegan") is True:
        df = df[df["is_vegan"] == True]

    if constraints.get("no_milk") is True:
        milk = df["contains_milk"] if "contains_milk" in df.columns else _compute_contains_milk(df)
        df = df[~milk.astype(bool)]

    caffeine_level = constraints.get("caffeine_level")
    if caffeine_level is not None and caffeine_level in CAFFEINE_RANGES:
        lo, hi = CAFFEINE_RANGES[caffeine_level]
        df = df[(df["caffeine_mg"] >= lo) & (df["caffeine_mg"] <= hi)]

    return df


if __name__ == "__main__":
    products = load_products("products.csv")

    test_constraints = {"category": "brewed", "max_sugar": 45, "caffeine_level": "high"}
    result = filter_products(test_constraints, products)

    print(f"Constraints: {test_constraints}")
    print(f"Products passing filter: {len(result)}")
    print(result[["product_id", "name", "caffeine_mg", "sugar_g"]].to_string(index=False))

    # Cross-check against TRAIN_001 which has the same constraints
    # Expected: BRW_001, BRW_002, BRW_003, BRW_005 (not BRW_004 decaf)
    expected = {"BRW_001", "BRW_002", "BRW_003", "BRW_005"}
    got = set(result["product_id"])
    print(f"\nExpected IDs : {sorted(expected)}")
    print(f"Got IDs      : {sorted(got)}")
    print(f"Match        : {got == expected}")
