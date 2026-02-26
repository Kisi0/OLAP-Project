"""
Global Retail Sales Dataset Generator
Generates 10,000 synthetic transactions for OLAP analysis.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

# ── Dimension values ────────────────────────────────────────────────────────
REGIONS = {
    "North America": ["United States", "Canada", "Mexico"],
    "Europe":        ["Germany", "France", "United Kingdom", "Italy", "Spain"],
    "Asia Pacific":  ["China", "Japan", "Australia", "India", "South Korea"],
    "Latin America": ["Brazil", "Argentina", "Chile", "Colombia"],
}

PRODUCTS = {
    "Electronics":     ["Laptops", "Smartphones", "Tablets", "Cameras", "Audio"],
    "Furniture":       ["Chairs", "Desks", "Shelves", "Sofas", "Tables"],
    "Office Supplies": ["Paper", "Pens", "Staplers", "Binders", "Notebooks"],
    "Clothing":        ["Shirts", "Pants", "Jackets", "Shoes", "Accessories"],
}

SEGMENTS = ["Consumer", "Corporate", "Home Office"]

# Price ranges per category
PRICE_RANGES = {
    "Electronics":     (50,  2500),
    "Furniture":       (80,  1500),
    "Office Supplies": (5,   200),
    "Clothing":        (15,  400),
}

MARGIN_RANGES = {
    "Electronics":     (0.10, 0.35),
    "Furniture":       (0.20, 0.50),
    "Office Supplies": (0.30, 0.60),
    "Clothing":        (0.25, 0.55),
}


def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def generate_dataset(n=10000):
    start_date = datetime(2022, 1, 1)
    end_date   = datetime(2024, 12, 31)

    rows = []
    for i in range(n):
        # Time
        order_date = random_date(start_date, end_date)
        year    = order_date.year
        quarter = f"Q{(order_date.month - 1) // 3 + 1}"
        month   = order_date.month
        month_name = order_date.strftime("%B")

        # Geography
        region  = random.choice(list(REGIONS.keys()))
        country = random.choice(REGIONS[region])

        # Product
        category    = random.choice(list(PRODUCTS.keys()))
        subcategory = random.choice(PRODUCTS[category])

        # Customer
        segment = random.choice(SEGMENTS)

        # Measures
        lo, hi   = PRICE_RANGES[category]
        unit_price = round(random.uniform(lo, hi), 2)
        quantity   = random.randint(1, 20)
        revenue    = round(unit_price * quantity, 2)

        mlo, mhi  = MARGIN_RANGES[category]
        margin     = random.uniform(mlo, mhi)
        cost       = round(revenue * (1 - margin), 2)
        profit     = round(revenue - cost, 2)
        profit_margin = round(profit / revenue * 100, 2)

        # Seasonal boost
        if quarter in ("Q4",) and category == "Electronics":
            revenue = round(revenue * 1.15, 2)
            profit  = round(revenue - cost, 2)
            profit_margin = round(profit / revenue * 100, 2)

        rows.append({
            "order_id":       f"ORD-{i+1:06d}",
            "order_date":     order_date.strftime("%Y-%m-%d"),
            "year":           year,
            "quarter":        quarter,
            "month":          month,
            "month_name":     month_name,
            "region":         region,
            "country":        country,
            "category":       category,
            "subcategory":    subcategory,
            "customer_segment": segment,
            "quantity":       quantity,
            "unit_price":     unit_price,
            "revenue":        revenue,
            "cost":           cost,
            "profit":         profit,
            "profit_margin":  profit_margin,
        })

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/global_retail_sales.csv", index=False)
    print(f"Dataset generated: {len(df):,} rows → data/global_retail_sales.csv")
    print(df.describe())
    return df


if __name__ == "__main__":
    generate_dataset()
