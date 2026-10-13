"""Page 3 — Basket Combos (market-basket / Apriori association rules)."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import SEGMENTS
from ui import SEG_COLORS, get_artifacts, hero, insights, kpi_row, show, style, table

A = get_artifacts()
P, share, bsize = A["pairs"], A["item_share"], A["basket_size"]
nice = lambda s: s.replace("_", " ")

hero("Basket Combos", "Which menu items are ordered together — overall and inside each segment", "🛒", ("#FF6B35", "#FFB703"))

kpi_row([("All baskets", f"{A['n_baskets']:,}", f"{A['orders']['item'].nunique()} menu items", "#F26B21")] +
        [(f"{seg} · basket size", f"{bsize[seg]:.2f}", "average items per order", SEG_COLORS[seg]) for seg in SEGMENTS])
st.write("")

sel = st.radio("Show combos for", ["All diners"] + SEGMENTS, horizontal=True)
pairs = P[sel].copy()
color = "#F26B21" if sel == "All diners" else SEG_COLORS[sel]

c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure(go.Bar(
        y=[nice(r) for r in pairs["rule"]], x=pairs["lift"], orientation="h", marker_color=color,
        text=[f"lift {l:.1f} · {c:.0%} of the time" for l, c in zip(pairs["lift"], pairs["confidence"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>lift %{x:.2f}<extra></extra>"))
    fig.add_vline(x=1, line_dash="dash", line_color="#6B7280", annotation_text="lift 1 = no association",
                  annotation_position="bottom right")
    fig.update_xaxes(range=[0, max(pairs["lift"].max(), 1.5) * 1.6])
    fig.update_yaxes(autorange="reversed")
    show(style(fig, f"Strongest item combos — {sel}", "Lift (× more often than by chance)",
               "If a diner orders …  →  they also order …", 380, legend=False))
with c2:
    t = pairs.assign(Rule=pairs["rule"].map(nice), Support=pairs["support"] * 100,
                     Confidence=pairs["confidence"] * 100)[["Rule", "Support", "Confidence", "lift"]]
    table(t, column_config={
        "Support": st.column_config.NumberColumn("Support (% of baskets)", format="%.1f"),
        "Confidence": st.column_config.ProgressColumn("Confidence (%)", min_value=0, max_value=100, format="%.0f%%"),
        "lift": st.column_config.NumberColumn("Lift", format="%.2f")})
    st.caption("Confidence = share of baskets with the first item that also contain the second.")

# ---- menu mix heatmap ---------------------------------------------------------------
order = share.max(axis=1).sort_values(ascending=False).index
z = share.loc[order] * 100
fig = go.Figure(go.Heatmap(
    z=z.values, x=SEGMENTS, y=[nice(i) for i in order],
    text=[[f"{v:.0f}%" if v >= 0.5 else "" for v in row] for row in z.values], texttemplate="%{text}",
    colorscale=[[0, "#F5F3FF"], [0.5, "#B197FC"], [1, "#6741D9"]],
    colorbar=dict(title="% of segment's items"), hovertemplate="%{y} in %{x}: %{z:.1f}%<extra></extra>"))
fig.update_yaxes(autorange="reversed")
show(style(fig, "Menu mix: how each segment's items split across the menu", "Segment", "Menu item", 520, legend=False, wrap=90))

# ---- key insights --------------------------------------------------------------------
stu_items = set(share.index[share["Budget students"] > 0])
other_items = set(share.index[(share[["At-risk families", "Premium foodies"]] > 0).any(axis=1)])
shared = sorted(stu_items & other_items)
top3 = P["All diners"].head(3)
fam_top, pre_top = P["At-risk families"].head(2), P["Premium foodies"].head(2)
stu_best = P["Budget students"]["lift"].max()
insights([
    "<b>Strongest overall combos:</b> " + "; ".join(f"{nice(r.rule)} (lift {r.lift:.1f})" for r in top3.itertuples())
    + " — promote each pair together.",
    f"<b>Overall lifts are inflated by segment mix:</b> menus barely overlap — Budget students order only "
    f"{len(stu_items)} items and share just {', '.join(nice(s) for s in shared)} with other segments.",
    "<b>Families:</b> " + "; ".join(f"{nice(r.rule)} (lift {r.lift:.1f})" for r in fam_top.itertuples())
    + " → build a 'family feast' bundle.",
    "<b>Premium foodies:</b> " + "; ".join(f"{nice(r.rule)} (lift {r.lift:.1f})" for r in pre_top.itertuples())
    + " → curated 'dinner for two' sets.",
    f"<b>Budget students:</b> best lift is only {stu_best:.2f} — they buy the same cheap staples regardless of basket, "
    f"so a fixed-price combo deal beats cross-selling.",
])
