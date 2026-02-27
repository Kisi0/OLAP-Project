"""
app.py – OLAP BI Assistant (Tier 2 – Builder)
Run: python -m streamlit run app.py
"""
import os
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
    
    # DEBUG: Shfaq pyetjen në terminal
    print(f"\n===== Processing query: {q} =====")

    # ── PIVOT OPERATION - Më i zgjeruar ──────────────────────────────────
    pivot_keywords = ['pivot', 'crosstab', 'cross tab', 'matrix', 'table', 'show by', 'breakdown by']
    if any(kw in q for kw in pivot_keywords):
        print("✓ Pivot detected!")
        
        # Dimensionet e mundshme
        dimensions = {
            'category': ['category', 'categories', 'product', 'products'],
            'region': ['region', 'regions'],
            'country': ['country', 'countries'],
            'year': ['year', 'years'],
            'quarter': ['quarter', 'quarters'],
            'month': ['month', 'months'],
            'customer_segment': ['segment', 'customer', 'segments']
        }
        
        # Gjej dimensionet e përmendura
        found_dims = []
        for dim, keywords in dimensions.items():
            if any(kw in q for kw in keywords):
                found_dims.append(dim)
                print(f"  - Found dimension: {dim}")
        
        # Nëse gjetëm të paktën 2 dimensione
        if len(found_dims) >= 2:
            rows = found_dims[0]
            cols = found_dims[1]
            
            # Cila matje?
            measure = 'revenue'
            if 'profit' in q:
                measure = 'profit'
            elif 'quantity' in q:
                measure = 'quantity'
            elif 'margin' in q:
                measure = 'profit_margin'
            
            print(f"  - Creating pivot: {rows} x {cols} = {measure}")
            
            try:
                # Krijo pivot
                pivot_data = data[[rows, cols, measure]].dropna()
                
                if len(pivot_data) == 0:
                    return "No data available for these dimensions."
                
                pivot_result = pd.pivot_table(
                    pivot_data,
                    index=rows,
                    columns=cols,
                    values=measure,
                    aggfunc='sum',
                    fill_value=0,
                    margins=True,
                    margins_name='Total'
                )
                
                # Format përgjigjen
                lines = [f"### Pivot Table: {rows.title()} by {cols.title()} ({measure})", ""]
                lines.append(pivot_result.to_markdown(floatfmt=",.2f"))
                
                # Insight
                if not pivot_result.empty and 'Total' not in pivot_result.index:
                    # Gjej vlerën më të lartë duke përjashtuar Total
                    data_only = pivot_result.drop('Total', errors='ignore')
                    if not data_only.empty:
                        max_val = data_only.max().max()
                        # Gjej pozicionin
                        for r in data_only.index:
                            for c in data_only.columns:
                                if data_only.loc[r, c] == max_val:
                                    lines.append(f"\n**💡 Key Insight:** Highest {measure} is **${max_val:,.0f}** "
                                               f"at **{r} / {c}**")
                                    break
                
                return "\n".join(lines)
                
            except Exception as e:
                return f"Error creating pivot: {str(e)}"
    
    # ── Drill-down ────────────────────────────────────────────────────────
    if any(kw in q for kw in ["drill", "break down", "by month", "by quarter", "by week"]):
        print("✓ Drill-down detected!")
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
        print("✓ YoY/Compare detected!")
        result = yoy_growth(data, 2023, 2024, group_by=["region"])
        op = "Compare / YoY Growth"

    # ── Top N ────────────────────────────────────────────────────────────
    elif "top" in q and ("countr" in q or "region" in q or "category" in q):
        print("✓ Top N detected!")
        dim = "country" if "countr" in q else ("region" if "region" in q else "category")
        measure = "profit" if "profit" in q else "revenue"
        result = top_n(data, dim, measure, n=5)
        op = f"Top 5 {dim.title()} by {measure.title()}"

    # ── Revenue share ─────────────────────────────────────────────────────
    elif "percentage" in q or "share" in q or "proportion" in q:
        print("✓ Revenue share detected!")
        dim = "category" if "category" in q else "region"
        result = revenue_share(data, dim)
        op = f"Revenue Share by {dim.title()}"

    # ── Monthly trend ─────────────────────────────────────────────────────
    elif "monthly" in q or "trend" in q:
        print("✓ Monthly trend detected!")
        yr = None
        for y in [2022, 2023, 2024]:
            if str(y) in q:
                yr = y
                break
        result = monthly_trend(data, yr)
        op = f"Monthly Trend {'(' + str(yr) + ')' if yr else '(All Years)'}"

    # ── Slice – single filter ─────────────────────────────────────────────
    elif "electronics" in q and "europe" in q:
        print("✓ Electronics in Europe detected!")
        sub = dice_data(data, category="Electronics", region="Europe")
        result = aggregate(sub, ["year", "quarter"], ["revenue", "profit"])
        op = "Dice – Electronics × Europe"

    elif "corporate" in q:
        print("✓ Corporate segment detected!")
        sub = slice_data(data, customer_segment="Corporate")
        result = aggregate(sub, ["year", "category"], ["revenue", "profit"])
        op = "Slice – Corporate Segment"

    # ── Default – by region ───────────────────────────────────────────────
    else:
        print("  - No specific operation detected, using default aggregate")
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
        "Pivot Category by Region": "Show pivot of category and region",
        "Pivot Category by Country": "Show pivot of category and country",
        "Pivot Region by Year": "Show pivot of region and year",
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
tab_chat, tab_dashboard, tab_olap, tab_data = st.tabs(
    ["💬 AI Chat", "📈 Dashboard", "📊 OLAP Explorer", "📋 Data"]
)

# ─────────────────────────────────────────────
# CHAT TAB
# ─────────────────────────────────────────────
with tab_chat:
    st.header("💬 OLAP AI Assistant")
    st.markdown("Ask any question about your data in plain English!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Pre-fill from sidebar
    if clicked and clicked not in [m["content"] for m in st.session_state.messages if m["role"]=="user"]:
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
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────
# DASHBOARD TAB
# ─────────────────────────────────────────────
with tab_dashboard:
    st.header("📈 Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"${filtered_df['revenue'].sum():,.0f}")
    with col2:
        st.metric("Total Profit", f"${filtered_df['profit'].sum():,.0f}")
    with col3:
        st.metric("Avg Margin", f"{filtered_df['profit_margin'].mean():.1f}%")
    with col4:
        st.metric("Transactions", f"{len(filtered_df):,}")
    
    st.divider()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Revenue by Region")
        reg_data = aggregate(filtered_df, ["region"], ["revenue"])
        fig = px.bar(reg_data, x="region", y="revenue", 
                     color="region", title="Revenue by Region")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("Revenue by Category")
        cat_data = aggregate(filtered_df, ["category"], ["revenue"])
        fig2 = px.pie(cat_data, names="category", values="revenue", 
                      title="Revenue by Category", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Monthly Revenue Trend")
    trend = filtered_df.groupby(['year', 'month'])['revenue'].sum().reset_index()
    trend['date'] = trend['year'].astype(str) + '-' + trend['month'].astype(str).str.zfill(2)
    fig3 = px.line(trend, x='date', y='revenue', title="Monthly Revenue Trend", markers=True)
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# OLAP EXPLORER TAB
# ─────────────────────────────────────────────
with tab_olap:
    st.header("📊 OLAP Explorer - Pivot Tables")
    
    st.subheader("Create Your Own Pivot Table")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get categorical columns for rows
        row_options = ["category", "region", "year", "quarter", "customer_segment", "country", "month"]
        rows = st.selectbox("Rows (Index)", row_options)
    
    with col2:
        # Get categorical columns for columns
        col_options = ["region", "category", "year", "quarter", "customer_segment", "country", "month"]
        # Remove the selected row from column options if present
        col_options = [c for c in col_options if c != rows]
        cols = st.selectbox("Columns", col_options, index=0 if col_options else 0)
    
    with col3:
        # Values to aggregate
        values = st.selectbox("Values", ["revenue", "profit", "quantity", "profit_margin"])
    
    agg_func = st.selectbox("Aggregation Function", ["sum", "mean", "count", "max", "min"])
    
    if st.button("Generate Pivot Table", type="primary"):
        try:
            # Create pivot table
            pivot_data = filtered_df[[rows, cols, values]].dropna()
            
            if len(pivot_data) == 0:
                st.warning("No data available for selected dimensions")
            else:
                pivot_result = pd.pivot_table(
                    pivot_data,
                    index=rows,
                    columns=cols,
                    values=values,
                    aggfunc=agg_func,
                    fill_value=0,
                    margins=True,
                    margins_name='Total'
                )
                
                st.success("Pivot table created successfully!")
                
                # Display as dataframe
                st.dataframe(pivot_result, use_container_width=True)
                
                # Visualize as heatmap (without Total row/col)
                viz_data = pivot_result.drop('Total', errors='ignore').drop('Total', axis=1, errors='ignore')
                if viz_data.size > 0:
                    fig = px.imshow(
                        viz_data,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="Blues",
                        title=f"{agg_func.title()} of {values} by {rows} and {cols}"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error creating pivot table: {str(e)}")
            st.write("Sample data:", filtered_df[[rows, cols, values]].head())

# ─────────────────────────────────────────────
# DATA TAB
# ─────────────────────────────────────────────
with tab_data:
    st.header("📋 Dataset Explorer")
    st.markdown(f"Showing **{len(filtered_df):,}** of **{len(df):,}** records")
    
    # Column info
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': filtered_df.columns,
        'Type': filtered_df.dtypes.values,
        'Unique Values': [filtered_df[c].nunique() for c in filtered_df.columns]
    })
    st.dataframe(col_info, use_container_width=True)
    
    # Show data sample
    st.subheader("Data Sample")
    st.dataframe(filtered_df.head(100), use_container_width=True)
    
    # Statistics
    st.subheader("Statistical Summary")
    st.dataframe(filtered_df[['revenue', 'profit', 'quantity', 'profit_margin']].describe(), use_container_width=True)
    
    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Filtered Data as CSV",
        csv,
        "filtered_data.csv",
        "text/csv",
        key='download-csv'
    )
