"""
app.py – OLAP BI Assistant (Tier 2 – Builder)
Run: python -m streamlit run app.py
"""
import os
import streamlit as st
import pandas as pd
import plotly.express as px

from data_utils import (
    load_data, slice_data, dice_data, aggregate,
    drill_down, roll_up, pivot_table, drill_through,
    yoy_growth, top_n, revenue_share, monthly_trend,
    dataset_summary,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="OLAP BI Assistant",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def get_data():
    path = "data/global_retail_sales.csv"
    if not os.path.exists(path):
        import subprocess
        subprocess.run(["python", "generate_dataset.py"], check=True)
    return load_data(path)


df = get_data()
summary = dataset_summary(df)


# ─────────────────────────────────────────────
# FUNKSIONET KRYESORE - PARA SE TË PËRDOREN
# ─────────────────────────────────────────────
def process_query(question: str, data: pd.DataFrame, api_key: str) -> str:
    """Route query to correct OLAP operation."""
    q = question.lower()

    # ── Drill-down ────────────────────────────────────────────────────────
    if any(kw in q for kw in ["drill", "break down", "by month", "by quarter", "by week"]):
        if "q4" in q and "month" in q:
            sub = dice_data(data, quarter="Q4")
            if "2024" in q:
                sub = dice_data(sub, year=2024)
            result = aggregate(sub, ["month", "month_name"], ["revenue", "profit"])
            result = result.sort_values("month")
            op = "Drill-Down (Q4 → Month)"
        elif "year" in q and ("quarter" in q or "q" in q):
            result = aggregate(data, ["year", "quarter"], ["revenue", "profit"])
            result = result.sort_values(["year", "quarter"])
            op = "Drill-Down (Year → Quarter)"
        else:
            result = aggregate(data, ["category", "subcategory"], ["revenue", "profit"])
            op = "Drill-Down (Category → Subcategory)"

    # ── YoY / Compare ────────────────────────────────────────────────────
    elif any(kw in q for kw in ["compare", "growth", "yoy", "2023 vs 2024", "year over"]):
        result = yoy_growth(data, 2023, 2024, group_by=["region"])
        op = "Compare / YoY Growth"

    # ── Top N ────────────────────────────────────────────────────────────
    elif "top" in q and ("countr" in q or "region" in q or "category" in q):
        dim = "country" if "countr" in q else ("region" if "region" in q else "category")
        measure = "profit" if "profit" in q else "revenue"
        result = top_n(data, dim, measure, n=5)
        op = f"Top 5 {dim.title()} by {measure.title()}"

    # ── Revenue share ─────────────────────────────────────────────────────
    elif "percentage" in q or "share" in q or "proportion" in q:
        dim = "category" if "category" in q else "region"
        result = revenue_share(data, dim)
        op = f"Revenue Share by {dim.title()}"

    # ── Monthly trend ─────────────────────────────────────────────────────
    elif "monthly" in q or "trend" in q:
        yr = None
        for y in [2022, 2023, 2024]:
            if str(y) in q:
                yr = y
                break
        result = monthly_trend(data, yr)
        op = f"Monthly Trend {'(' + str(yr) + ')' if yr else '(All Years)'}"

    # ── Slice – single filter ─────────────────────────────────────────────
    elif "electronics" in q and "europe" in q:
        sub = dice_data(data, category="Electronics", region="Europe")
        result = aggregate(sub, ["year", "quarter"], ["revenue", "profit"])
        op = "Dice – Electronics × Europe"

    elif "corporate" in q:
        sub = slice_data(data, customer_segment="Corporate")
        result = aggregate(sub, ["year", "category"], ["revenue", "profit"])
        op = "Slice – Corporate Segment"

    # ── Default – by region ───────────────────────────────────────────────
    else:
        result = aggregate(data, ["region"], ["revenue", "profit", "quantity"])
        op = "Aggregate – Revenue by Region"

    return _format_response(op, result)


def _format_response(op: str, result: pd.DataFrame) -> str:
    lines = [f"### Operation: {op}", ""]
    
    # Table
    if len(result) <= 30:
        lines.append(result.to_markdown(index=False, floatfmt=",.2f"))
    else:
        lines.append(result.head(20).to_markdown(index=False, floatfmt=",.2f"))
        lines.append(f"*… {len(result)-20} more rows*")
    
    # Insight
    if "revenue" in result.columns:
        top_row = result.loc[result["revenue"].idxmax()]
        top_label = top_row.iloc[0]
        top_val = top_row["revenue"]
        lines.append(f"\n**💡 Key Insight:** **{top_label}** leads with "
                     f"**${top_val:,.0f}** in revenue.")
    
    # Follow-ups
    lines += [
        "\n**Suggested Follow-ups:**",
        "- Drill deeper into the top performer by sub-dimension",
        "- Compare this metric across customer segments",
        "- Pivot the view to see time periods as columns",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📊 OLAP Assistant")
    st.caption("Global Retail Sales 2022-2024")
    st.divider()
    
    st.subheader("🔍 Quick Filters")
    sel_year = st.multiselect("Year", [2022, 2023, 2024], default=[2022, 2023, 2024])
    sel_region = st.multiselect("Region", summary["regions"], default=summary["regions"])
    sel_cat = st.multiselect("Category", summary["categories"], default=summary["categories"])
    
    st.divider()
    st.subheader("⚡ Quick OLAP Queries")
    quick_ops = {
        "Revenue by Region": "Show total revenue by region",
        "YoY Growth (2023 vs 2024)": "Compare 2023 vs 2024 revenue by region",
        "Drill Q4 2024 by Month": "Drill down Q4 2024 by month",
        "Top 5 Countries by Profit": "Show top 5 countries by profit",
    }
    clicked = None
    for label, query in quick_ops.items():
        if st.button(label, use_container_width=True):
            clicked = query
    
    st.divider()
    api_key = st.text_input("Anthropic API Key (optional)", type="password")

# Filter data
filtered_df = df[
    (df["year"].isin(sel_year)) &
    (df["region"].isin(sel_region)) &
    (df["category"].isin(sel_cat))
]

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_chat, tab_dashboard, tab_data = st.tabs(
    ["💬 AI Chat", "📈 Dashboard", "📋 Data"]
)

# ─────────────────────────────────────────────
# CHAT TAB
# ─────────────────────────────────────────────
with tab_chat:
    st.header("💬 OLAP AI Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Pre-fill from sidebar
    if clicked:
        st.session_state.messages.append({"role": "user", "content": clicked})
    
    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Handle new input
    user_input = st.chat_input("Ask an OLAP question…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Analysing…"):
                response = process_query(user_input, filtered_df, api_key)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ─────────────────────────────────────────────
# DASHBOARD TAB (thjeshtuar për testim)
# ─────────────────────────────────────────────
with tab_dashboard:
    st.header("Executive Dashboard")
    total_rev = filtered_df["revenue"].sum()
    st.metric("Total Revenue", f"${total_rev:,.0f}")

# ─────────────────────────────────────────────
# DATA TAB
# ─────────────────────────────────────────────
with tab_data:
    st.header("Dataset View")
    st.dataframe(filtered_df.head(100))
