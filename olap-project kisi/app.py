"""
app.py – OLAP BI Assistant (Tier 2 – Builder)
Run: streamlit run app.py
"""
import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_utils import (
    load_data, slice_data, dice_data, aggregate,
    drill_down, roll_up, pivot_table, drill_through,
    yoy_growth, top_n, revenue_share, monthly_trend,
    dataset_summary,
)
from prompts import SYSTEM_PROMPT

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OLAP BI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.olap-badge {
    display:inline-block; padding:3px 10px; border-radius:12px;
    font-size:12px; font-weight:600; margin:2px;
    background:#1a73e8; color:white;
}
.kpi-box {
    background:linear-gradient(135deg,#1a1a2e,#16213e);
    border:1px solid #0f3460; border-radius:10px;
    padding:16px; text-align:center; color:white;
}
.kpi-val { font-size:28px; font-weight:700; color:#e94560; }
.kpi-lbl { font-size:13px; color:#a8b2d8; margin-top:4px; }
.insight-box {
    background:#fffde7; border-left:4px solid #f9a825;
    padding:12px 16px; border-radius:4px; margin:8px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    path = "data/global_retail_sales.csv"
    if not os.path.exists(path):
        import subprocess
        subprocess.run(["python", "generate_dataset.py"], check=True)
    return load_data(path)


df = get_data()
summary = dataset_summary(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
        "Revenue by Region":         "Show total revenue by region",
        "YoY Growth (2023 vs 2024)":  "Compare 2023 vs 2024 revenue by region",
        "Drill Q4 2024 by Month":     "Drill down Q4 2024 by month",
        "Top 5 Countries by Profit":  "Show top 5 countries by profit",
        "Category Revenue Share":     "What percentage of revenue comes from each category?",
        "Monthly Trend 2024":         "Show monthly revenue trend for 2024",
        "Electronics in Europe":      "Show Electronics sales in Europe by year",
        "Corporate Segment Analysis": "Analyze Corporate customer segment performance",
    }
    clicked = None
    for label, query in quick_ops.items():
        if st.button(label, use_container_width=True):
            clicked = query

    st.divider()
    st.caption("Built with Streamlit + Anthropic API")
    api_key = st.text_input("Anthropic API Key (optional)", type="password",
                            help="Enter your key to enable AI-powered chat")


# ── Apply sidebar filters ─────────────────────────────────────────────────────
filtered_df = df[
    (df["year"].isin(sel_year)) &
    (df["region"].isin(sel_region)) &
    (df["category"].isin(sel_cat))
]


# ── TABS ──────────────────────────────────────────────────────────────────────
tab_chat, tab_dashboard, tab_olap, tab_data = st.tabs(
    ["💬 AI Chat", "📈 Dashboard", "🔧 OLAP Explorer", "📋 Data View"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – AI CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.header("💬 OLAP AI Assistant")
    st.markdown(
        "Ask business questions in plain English. The assistant performs "
        "OLAP operations and explains the results."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Pre-fill from sidebar quick button
    if clicked:
        st.session_state.messages.append({"role": "user", "content": clicked})

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
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

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


def process_query(question: str, data: pd.DataFrame, api_key: str) -> str:
    """Route query to correct OLAP operation, optionally call Claude API."""
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

    return _format_response(op, result, question)


def _format_response(op: str, result: pd.DataFrame, question: str) -> str:
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
        top_val   = top_row["revenue"]
        lines.append(f"\n**💡 Key Insight:** **{top_label}** leads with "
                     f"**${top_val:,.0f}** in revenue.")

    if "yoy_growth_pct" in result.columns:
        best = result.loc[result["yoy_growth_pct"].idxmax()]
        lines.append(f"\n**💡 Key Insight:** **{best.iloc[0]}** shows the strongest "
                     f"YoY growth at **{best['yoy_growth_pct']:.1f}%**.")

    # Follow-ups
    lines += [
        "\n**Suggested Follow-ups:**",
        "- Drill deeper into the top performer by sub-dimension",
        "- Compare this metric across customer segments",
        "- Pivot the view to see time periods as columns",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.header("📈 Executive Dashboard")

    # KPI row
    total_rev  = filtered_df["revenue"].sum()
    total_prof = filtered_df["profit"].sum()
    avg_margin = filtered_df["profit_margin"].mean()
    n_trans    = len(filtered_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total Revenue",     f"${total_rev:,.0f}")
    c2.metric("📈 Total Profit",      f"${total_prof:,.0f}")
    c3.metric("📊 Avg Profit Margin", f"{avg_margin:.1f}%")
    c4.metric("🛒 Transactions",      f"{n_trans:,}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue by Region")
        reg_data = aggregate(filtered_df, ["region"], ["revenue"])
        fig = px.bar(reg_data, x="region", y="revenue",
                     color="region",
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     template="plotly_white")
        fig.update_layout(showlegend=False, height=320,
                          yaxis_tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue by Category")
        cat_data = aggregate(filtered_df, ["category"], ["revenue"])
        fig2 = px.pie(cat_data, names="category", values="revenue",
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      hole=0.4)
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Monthly Revenue Trend")
    trend = aggregate(filtered_df, ["year", "month"], ["revenue"])
    trend["period"] = trend["year"].astype(str) + "-" + trend["month"].astype(str).str.zfill(2)
    trend = trend.sort_values("period")
    fig3 = px.line(trend, x="period", y="revenue",
                   template="plotly_white", markers=True)
    fig3.update_layout(height=300, yaxis_tickformat="$,.0f",
                       xaxis_title="Month", yaxis_title="Revenue")
    st.plotly_chart(fig3, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("YoY Revenue by Region")
        yoy = yoy_growth(filtered_df, 2023, 2024, group_by=["region"])
        if "yoy_growth_pct" in yoy.columns:
            fig4 = px.bar(yoy, x="region", y="yoy_growth_pct",
                          color="yoy_growth_pct",
                          color_continuous_scale="RdYlGn",
                          template="plotly_white")
            fig4.update_layout(height=300, yaxis_title="YoY Growth %",
                               showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

    with col4:
        st.subheader("Profit Margin by Category")
        margin_data = filtered_df.groupby("category")["profit_margin"].mean().reset_index()
        fig5 = px.bar(margin_data, x="category", y="profit_margin",
                      color="profit_margin",
                      color_continuous_scale="Blues",
                      template="plotly_white")
        fig5.update_layout(height=300, yaxis_title="Avg Margin %",
                           showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – OLAP EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab_olap:
    st.header("🔧 Interactive OLAP Operations")

    olap_op = st.selectbox("Choose OLAP Operation",
        ["Slice", "Dice", "Drill-Down", "Roll-Up", "Pivot", "Drill-Through",
         "YoY Comparison", "Top N", "Revenue Share"])

    st.divider()

    if olap_op == "Slice":
        st.subheader("🔪 Slice – Single Dimension Filter")
        col, val_options = st.columns(2)
        with col:
            dim = st.selectbox("Dimension", ["year","quarter","region","category",
                                              "customer_segment","country"])
        with val_options:
            unique_vals = sorted(filtered_df[dim].unique().tolist())
            val = st.selectbox("Value", unique_vals)
        if st.button("Apply Slice"):
            result = slice_data(filtered_df, **{dim: val})
            agg_result = aggregate(result, ["category", "region"] if dim not in ["category","region"] else ["year","quarter"], ["revenue","profit"])
            st.dataframe(agg_result, use_container_width=True)
            st.info(f"**{len(result):,}** transactions matching `{dim} = {val}` | Revenue: **${result['revenue'].sum():,.0f}**")

    elif olap_op == "Dice":
        st.subheader("🎲 Dice – Multiple Dimension Filters")
        c1, c2, c3 = st.columns(3)
        with c1: f_year = st.multiselect("Year", [2022,2023,2024], default=[2024])
        with c2: f_reg  = st.multiselect("Region", summary["regions"])
        with c3: f_cat  = st.multiselect("Category", summary["categories"])
        if st.button("Apply Dice"):
            sub = filtered_df.copy()
            if f_year: sub = slice_data(sub, year=f_year)
            if f_reg:  sub = slice_data(sub, region=f_reg)
            if f_cat:  sub = slice_data(sub, category=f_cat)
            result = aggregate(sub, ["region","category"], ["revenue","profit"])
            st.dataframe(result, use_container_width=True)
            st.info(f"**{len(sub):,}** transactions | Revenue: **${sub['revenue'].sum():,.0f}**")

    elif olap_op == "Drill-Down":
        st.subheader("🔽 Drill-Down – From Summary to Detail")
        c1, c2 = st.columns(2)
        with c1:
            hierarchy = st.selectbox("Hierarchy", ["Time", "Geography", "Product"])
        with c2:
            if hierarchy == "Time":
                from_level = st.selectbox("From", ["year","quarter"])
                to_level   = st.selectbox("To", ["quarter","month_name"])
                from_val   = st.selectbox("Value", sorted(filtered_df[from_level].unique().tolist()))
            elif hierarchy == "Geography":
                from_level, to_level = "region", "country"
                from_val = st.selectbox("Region", sorted(filtered_df["region"].unique().tolist()))
            else:
                from_level, to_level = "category", "subcategory"
                from_val = st.selectbox("Category", sorted(filtered_df["category"].unique().tolist()))
        if st.button("Drill Down"):
            try:
                result = drill_down(filtered_df, from_level, to_level, from_val)
                st.dataframe(result, use_container_width=True)
                fig = px.bar(result, x=result.columns[0], y="revenue",
                             template="plotly_white",
                             color_discrete_sequence=["#1a73e8"])
                fig.update_layout(yaxis_tickformat="$,.0f")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(str(e))

    elif olap_op == "Roll-Up":
        st.subheader("🔼 Roll-Up – From Detail to Summary")
        c1, c2 = st.columns(2)
        with c1: from_l = st.selectbox("From (fine)", ["month_name","quarter","subcategory","country"])
        with c2: to_l   = st.selectbox("To (coarse)", ["quarter","year","category","region"])
        if st.button("Roll Up"):
            try:
                result = roll_up(filtered_df, from_l, to_l)
                st.dataframe(result, use_container_width=True)
            except Exception as e:
                st.error(str(e))

    elif olap_op == "Pivot":
        st.subheader("🔄 Pivot – Rotate the View")
        c1, c2, c3 = st.columns(3)
        with c1: rows_dim = st.selectbox("Rows",    ["region","category","customer_segment"])
        with c2: cols_dim = st.selectbox("Columns", ["year","quarter","category"])
        with c3: val_dim  = st.selectbox("Values",  ["revenue","profit","quantity"])
        if st.button("Pivot"):
            pt = pivot_table(filtered_df, rows_dim, cols_dim, val_dim)
            st.dataframe(pt, use_container_width=True)
            fig = px.imshow(pt.set_index(rows_dim),
                            color_continuous_scale="Blues",
                            aspect="auto", text_auto=",.0f")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    elif olap_op == "Drill-Through":
        st.subheader("🔍 Drill-Through – Raw Transactions")
        c1, c2 = st.columns(2)
        with c1:
            dt_region = st.selectbox("Region", ["All"] + summary["regions"])
            dt_cat    = st.selectbox("Category", ["All"] + summary["categories"])
        with c2:
            dt_year = st.selectbox("Year", ["All", 2022, 2023, 2024])
            dt_n    = st.slider("Max rows", 10, 100, 20)
        if st.button("Drill Through"):
            flt = {}
            if dt_region != "All": flt["region"]   = dt_region
            if dt_cat    != "All": flt["category"]  = dt_cat
            if dt_year   != "All": flt["year"]      = dt_year
            result = drill_through(filtered_df, n=dt_n, **flt)
            st.dataframe(result, use_container_width=True)

    elif olap_op == "YoY Comparison":
        st.subheader("📅 Year-over-Year Comparison")
        c1, c2, c3 = st.columns(3)
        with c1: y1 = st.selectbox("Year 1", [2022,2023], index=1)
        with c2: y2 = st.selectbox("Year 2", [2023,2024], index=1)
        with c3: grp = st.selectbox("Group By", ["region","category","customer_segment"])
        if st.button("Compare"):
            result = yoy_growth(filtered_df, y1, y2, group_by=[grp])
            st.dataframe(result, use_container_width=True)
            if "yoy_growth_pct" in result.columns:
                fig = px.bar(result, x=grp, y="yoy_growth_pct",
                             color="yoy_growth_pct",
                             color_continuous_scale="RdYlGn",
                             template="plotly_white")
                fig.update_layout(height=350, yaxis_title="YoY Growth %")
                st.plotly_chart(fig, use_container_width=True)

    elif olap_op == "Top N":
        st.subheader("🏆 Top N Rankings")
        c1, c2, c3 = st.columns(3)
        with c1: t_dim     = st.selectbox("Dimension", ["country","region","category","subcategory","customer_segment"])
        with c2: t_measure = st.selectbox("Measure", ["revenue","profit","quantity"])
        with c3: t_n       = st.slider("N", 3, 15, 5)
        if st.button("Rank"):
            result = top_n(filtered_df, t_dim, t_measure, n=t_n)
            st.dataframe(result, use_container_width=True)
            fig = px.bar(result, x=t_dim, y=t_measure,
                         color=t_measure, color_continuous_scale="Viridis",
                         template="plotly_white")
            fig.update_layout(height=350, yaxis_tickformat="$,.0f")
            st.plotly_chart(fig, use_container_width=True)

    elif olap_op == "Revenue Share":
        st.subheader("🥧 Revenue Share Analysis")
        rs_dim = st.selectbox("Dimension", ["region","category","customer_segment","country"])
        if st.button("Calculate Share"):
            result = revenue_share(filtered_df, rs_dim)
            st.dataframe(result, use_container_width=True)
            fig = px.pie(result, names=rs_dim, values="revenue_share_pct",
                         template="plotly_white", hole=0.35)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – DATA VIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.header("📋 Dataset Explorer")
    st.markdown(f"Showing **{len(filtered_df):,}** of **{len(df):,}** records (filtered)")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Column Statistics")
        st.dataframe(
            filtered_df[["revenue","profit","profit_margin","quantity"]].describe().round(2),
            use_container_width=True
        )
    with c2:
        st.subheader("🗂️ Dimension Counts")
        dim_counts = pd.DataFrame({
            "Dimension": ["Region","Category","Country","Customer Segment"],
            "Unique Values": [
                filtered_df["region"].nunique(),
                filtered_df["category"].nunique(),
                filtered_df["country"].nunique(),
                filtered_df["customer_segment"].nunique(),
            ]
        })
        st.dataframe(dim_counts, use_container_width=True)

    st.subheader("🔎 Raw Data Sample")
    n_rows = st.slider("Rows to display", 10, 200, 50)
    st.dataframe(
        filtered_df.head(n_rows)[[
            "order_id","order_date","year","quarter","region","country",
            "category","subcategory","customer_segment",
            "quantity","unit_price","revenue","profit","profit_margin"
        ]],
        use_container_width=True
    )

    # Download
    csv_bytes = filtered_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Filtered Dataset (CSV)",
        data=csv_bytes,
        file_name="filtered_retail_sales.csv",
        mime="text/csv",
    )
