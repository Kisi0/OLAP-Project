# OLAP BI Assistant — Capstone Project (Tier 2: Builder)

The OLAP AI Assistant successfully processes all dataset-related queries through natural language, as demonstrated here with a pivot table request showing complete regional analysis with automated insights. Whether users ask for aggregations, drill-downs, trend analysis, comparisons, top performers, revenue shares, or custom pivot tables - the system understands and executes the correct OLAP operation every time.


PDF:
[OLAP_AI_ASSISTANT.pdf](https://github.com/user-attachments/files/25615436/OLAP_AI_ASSISTANT.pdf)


## What's in this package

| File | Description |
|------|-------------|
| `OLAP_Project_Complete_Guide.docx` | 40-page Word document — full project guide |
| `interactive_dashboard.html` | Standalone interactive dashboard (no install needed) |
| `app.py` | Main Streamlit web application |
| `data_utils.py` | OLAP operations library (Slice, Dice, Drill-Down, etc.) |
| `prompts.py` | AI system prompt and templates |
| `generate_dataset.py` | Creates the 10,000-row dataset |
| `requirements.txt` | Python dependencies |
| `data/global_retail_sales.csv` | Pre-generated dataset |

---

## Quick Start (HTML Dashboard — no install)

1. Open `interactive_dashboard.html` in any browser
2. Explore all 4 sections: Dashboard, OLAP Explorer, AI Chat, Data View
3. No Python, no server, no API key needed

---

## Streamlit App (full-featured)

```bash
pip install -r requirements.txt
python generate_dataset.py   # creates data/global_retail_sales.csv
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## 📊 OLAP Operations Covered

- **Slice** — Single dimension filter
- **Dice** — Multiple dimension filters  
- **Drill-Down** — Year→Quarter→Month, Region→Country, Category→Subcategory
- **Roll-Up** — Aggregate to coarser levels
- **Pivot** — Rotate dimension perspective
- **Drill-Through** — View raw transactions
- **YoY Comparison** — Year-over-year growth
- **Top N** — Rankings by any measure

---

## Dataset

- 10,000 transactions | 2022–2024 | 4 Regions | 4 Categories
- Total Revenue: $64,630,746 | Avg Margin: 36.4%
