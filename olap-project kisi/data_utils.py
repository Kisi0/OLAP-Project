"""
data_utils.py – OLAP helper functions (Slice, Dice, Drill-Down, Roll-Up, Pivot, Drill-Through)
"""
import pandas as pd
import numpy as np


# ── Load & cache ──────────────────────────────────────────────────────────────
def load_data(path: str = "data/global_retail_sales.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date"])
    return df


# ── 1. SLICE – single dimension filter ────────────────────────────────────────
def slice_data(df: pd.DataFrame, **filters) -> pd.DataFrame:
    """
    Apply a single-dimension filter.
    Example: slice_data(df, year=2024)
             slice_data(df, quarter="Q4")
             slice_data(df, region="Europe")
    """
    result = df.copy()
    for col, val in filters.items():
        if isinstance(val, (list, tuple)):
            result = result[result[col].isin(val)]
        else:
            result = result[result[col] == val]
    return result


# ── 2. DICE – multi-dimension filter ─────────────────────────────────────────
def dice_data(df: pd.DataFrame, **filters) -> pd.DataFrame:
    """
    Apply multiple dimension filters simultaneously.
    Example: dice_data(df, year=2024, region="Europe", category="Electronics")
    """
    return slice_data(df, **filters)


# ── 3. GROUP & SUMMARIZE ──────────────────────────────────────────────────────
def aggregate(df: pd.DataFrame,
              group_by: list,
              measures: list = None,
              agg_funcs: dict = None) -> pd.DataFrame:
    """
    Aggregate measures over group_by dimensions.
    Default: sum revenue, profit; mean profit_margin; sum quantity.
    """
    if measures is None:
        measures = ["revenue", "profit", "quantity"]
    if agg_funcs is None:
        agg_funcs = {m: "sum" for m in measures}
        if "profit_margin" in df.columns:
            agg_funcs["profit_margin"] = "mean"
    cols = group_by + [m for m in agg_funcs if m in df.columns]
    result = df[cols].groupby(group_by).agg(agg_funcs).reset_index()
    for m in ["revenue", "profit"]:
        if m in result.columns:
            result[m] = result[m].round(2)
    if "profit_margin" in result.columns:
        result["profit_margin"] = result["profit_margin"].round(2)
    return result


# ── 4. DRILL-DOWN ─────────────────────────────────────────────────────────────
def drill_down(df: pd.DataFrame,
               from_level: str,
               to_level: str,
               from_value=None,
               measure: str = "revenue") -> pd.DataFrame:
    """
    Navigate from a coarser level to a finer one.
    Time:       year → quarter → month
    Geography:  region → country
    Product:    category → subcategory
    """
    time_hierarchy   = ["year", "quarter", "month_name", "month"]
    geo_hierarchy    = ["region", "country"]
    prod_hierarchy   = ["category", "subcategory"]

    def find_hierarchy(level):
        for h in [time_hierarchy, geo_hierarchy, prod_hierarchy]:
            if level in h:
                return h
        raise ValueError(f"Unknown level: {level}")

    hierarchy = find_hierarchy(from_level)
    result = df.copy()
    if from_value is not None:
        result = result[result[from_level] == from_value]

    from_idx = hierarchy.index(from_level)
    to_idx   = hierarchy.index(to_level)
    if to_idx <= from_idx:
        raise ValueError(f"'{to_level}' is not finer than '{from_level}'")

    group_cols = hierarchy[from_idx + 1: to_idx + 1]
    return aggregate(result, group_cols, [measure])


# ── 5. ROLL-UP ────────────────────────────────────────────────────────────────
def roll_up(df: pd.DataFrame,
            from_level: str,
            to_level: str,
            measure: str = "revenue") -> pd.DataFrame:
    """
    Aggregate from fine to coarse level.
    Example: roll_up(df, "month_name", "quarter")
    """
    time_hierarchy  = ["year", "quarter", "month_name"]
    geo_hierarchy   = ["region", "country"]
    prod_hierarchy  = ["category", "subcategory"]

    def find_hierarchy(level):
        for h in [time_hierarchy, geo_hierarchy, prod_hierarchy]:
            if level in h:
                return h
        raise ValueError(f"Unknown level: {level}")

    hierarchy = find_hierarchy(from_level)
    to_idx    = hierarchy.index(to_level)
    group_cols = hierarchy[:to_idx + 1]
    return aggregate(df, group_cols, [measure])


# ── 6. PIVOT ──────────────────────────────────────────────────────────────────
def pivot_table(df: pd.DataFrame,
                rows: str,
                columns: str,
                values: str = "revenue",
                aggfunc: str = "sum") -> pd.DataFrame:
    """
    Rotate the perspective. Example: pivot_table(df, "region", "year", "revenue")
    """
    pt = df.pivot_table(index=rows, columns=columns, values=values,
                        aggfunc=aggfunc, fill_value=0)
    pt.columns = [str(c) for c in pt.columns]
    return pt.reset_index()


# ── 7. DRILL-THROUGH ──────────────────────────────────────────────────────────
def drill_through(df: pd.DataFrame, n: int = 20, **filters) -> pd.DataFrame:
    """
    Return underlying raw transactions matching filters.
    """
    result = slice_data(df, **filters)
    cols = ["order_id", "order_date", "region", "country",
            "category", "subcategory", "customer_segment",
            "quantity", "unit_price", "revenue", "profit"]
    return result[cols].head(n)


# ── 8. KPI HELPERS ────────────────────────────────────────────────────────────
def yoy_growth(df: pd.DataFrame,
               year1: int,
               year2: int,
               group_by: list = None,
               measure: str = "revenue") -> pd.DataFrame:
    """Year-over-year growth between two years."""
    if group_by is None:
        group_by = []

    def agg(year):
        sub = df[df["year"] == year]
        if group_by:
            return aggregate(sub, group_by, [measure])
        else:
            total = sub[measure].sum()
            return pd.DataFrame({"year": [year], measure: [round(total, 2)]})

    a = agg(year1).rename(columns={measure: f"{measure}_{year1}"})
    b = agg(year2).rename(columns={measure: f"{measure}_{year2}"})

    if group_by:
        merged = a.merge(b, on=group_by)
    else:
        merged = pd.concat([a, b], axis=1)

    col1, col2 = f"{measure}_{year1}", f"{measure}_{year2}"
    merged["yoy_growth_pct"] = (
        (merged[col2] - merged[col1]) / merged[col1].replace(0, np.nan) * 100
    ).round(2)
    return merged


def top_n(df: pd.DataFrame,
          group_by: str,
          measure: str = "revenue",
          n: int = 5,
          ascending: bool = False) -> pd.DataFrame:
    """Return top / bottom N by measure."""
    agg_df = aggregate(df, [group_by], [measure])
    return agg_df.sort_values(measure, ascending=ascending).head(n)


def revenue_share(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Percentage share of total revenue."""
    agg_df = aggregate(df, [group_by], ["revenue"])
    total  = agg_df["revenue"].sum()
    agg_df["revenue_share_pct"] = (agg_df["revenue"] / total * 100).round(2)
    return agg_df.sort_values("revenue", ascending=False)


def monthly_trend(df: pd.DataFrame, year: int = None) -> pd.DataFrame:
    """Monthly revenue trend (optionally filtered by year)."""
    sub = df if year is None else df[df["year"] == year]
    result = aggregate(sub, ["year", "month", "month_name"], ["revenue", "profit"])
    return result.sort_values(["year", "month"])


# ── 9. DATASET SUMMARY ────────────────────────────────────────────────────────
def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "total_records":    len(df),
        "total_revenue":    round(df["revenue"].sum(), 2),
        "total_profit":     round(df["profit"].sum(), 2),
        "avg_profit_margin":round(df["profit_margin"].mean(), 2),
        "date_range":       f"{df['order_date'].min().date()} → {df['order_date'].max().date()}",
        "regions":          sorted(df["region"].unique().tolist()),
        "categories":       sorted(df["category"].unique().tolist()),
        "years":            sorted(df["year"].unique().tolist()),
    }
