"""
core.py — QuickBite data, models and analytics.

Pure Python (no Streamlit import) so it can be tested or reused from a notebook.
`build_artifacts()` reproduces the whole case study from data/diners.csv and data/orders.csv:

    Task 1  K-Means segmentation          -> segment names, profiles, k selection table
    Task 2  Apriori basket analysis       -> rules overall and per segment
    Task 3  Random-forest reorder model   -> accuracy, confusion matrix, risk thresholds
    Bonus   Lifetime-spend regression     -> extension to the brief (not required by it)
    Task 4  Models for the live simulator -> scaler, kmeans, clf, reco (same keys as the brief)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_absolute_error, r2_score, silhouette_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# The five behaviour features used for clustering (order matters).
FEATURES = ["orders_per_month", "avg_order_value", "avg_items_per_order",
            "days_since_last_order", "discount_rate"]
CLF_FEATURES = FEATURES + ["age", "total_spend"]      # reorder classifier inputs
REG_FEATURES = FEATURES + ["age"]                     # spend regression inputs (no leakage of the target)

SEGMENTS = ["Budget students", "At-risk families", "Premium foodies"]
NICE = {
    "orders_per_month": "Orders per month",
    "avg_order_value": "Avg order value (AED)",
    "avg_items_per_order": "Items per order",
    "days_since_last_order": "Days since last order",
    "discount_rate": "Discount rate",
    "age": "Age",
    "total_spend": "Total spend (AED)",
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    diners = pd.read_csv(DATA_DIR / "diners.csv").drop_duplicates()
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    if set(diners["diner_id"]) != set(orders["diner_id"]):
        raise ValueError("diner_id values do not match between diners.csv and orders.csv")
    return diners, orders


# ----------------------------------------------------------------------------
# Association rules helpers
# ----------------------------------------------------------------------------
def mine_rules(basket_df: pd.DataFrame, min_support: float = 0.03, min_lift: float = 1.0) -> pd.DataFrame:
    freq = apriori(basket_df, min_support=min_support, use_colnames=True)
    try:
        rules = association_rules(freq, metric="lift", min_threshold=min_lift,
                                  num_itemsets=len(basket_df))
    except TypeError:                      # older mlxtend has no num_itemsets argument
        rules = association_rules(freq, metric="lift", min_threshold=min_lift)
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return rules.sort_values(["lift", "confidence"], ascending=False)


def top_pairs(rules: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """One-item -> one-item rules. A->B and B->A share the same lift, so keep the direction
    with the higher confidence (the more useful 'if they order A, they also order B')."""
    cols = ["rule", "antecedents", "consequents", "support", "confidence", "lift"]
    if rules.empty:
        return pd.DataFrame(columns=cols)
    r = rules[~rules["antecedents"].str.contains(",") & ~rules["consequents"].str.contains(",")].copy()
    r["pair"] = [tuple(sorted(p)) for p in zip(r["antecedents"], r["consequents"])]
    r["lift_r"] = r["lift"].round(6)
    r = r.sort_values(["lift_r", "confidence"], ascending=False).drop_duplicates("pair")
    r["rule"] = r["antecedents"] + " → " + r["consequents"]
    return r[cols].head(n).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Build everything
# ----------------------------------------------------------------------------
def build_artifacts() -> dict:
    diners, orders = load_data()
    A: dict = {"features": FEATURES, "clf_features": CLF_FEATURES, "reg_features": REG_FEATURES}

    # ---- Task 1: segmentation ------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(diners[FEATURES])

    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X_scaled, km.labels_)})
    k_table = pd.DataFrame(rows)

    k_final = 3
    kmeans = KMeans(n_clusters=k_final, n_init=10, random_state=42).fit(X_scaled)
    diners["cluster"] = kmeans.labels_

    prof = diners.groupby("cluster")[FEATURES].mean()
    budget_id = prof["orders_per_month"].idxmax()          # most frequent orderers
    family_id = prof["avg_items_per_order"].idxmax()       # biggest baskets
    premium_id = ({0, 1, 2} - {budget_id, family_id}).pop()
    segment_names = {int(budget_id): "Budget students",
                     int(family_id): "At-risk families",
                     int(premium_id): "Premium foodies"}
    diners["segment"] = diners["cluster"].map(segment_names)

    summary = diners.groupby("segment").agg(
        diners=("diner_id", "count"),
        reorder_rate=("reordered", "mean"),
        orders_per_month=("orders_per_month", "mean"),
        avg_order_value=("avg_order_value", "mean"),
        avg_items_per_order=("avg_items_per_order", "mean"),
        days_since_last_order=("days_since_last_order", "mean"),
        discount_rate=("discount_rate", "mean"),
        age=("age", "mean"),
        total_spend_mean=("total_spend", "mean"),
        total_spend_sum=("total_spend", "sum"),
    ).reindex(SEGMENTS)
    spend_by_outcome = (diners.assign(outcome=np.where(diners["reordered"] == 1, "Reordered", "Churned"))
                        .groupby(["segment", "outcome"])["total_spend"].sum().unstack(fill_value=0)
                        .reindex(SEGMENTS))
    ranges = diners[FEATURES].agg(["min", "max"])
    A.update(diners=diners, orders=orders, scaler=scaler, kmeans=kmeans, segment_names=segment_names,
             k_table=k_table, k_final=k_final, segment_summary=summary,
             spend_by_outcome=spend_by_outcome, feature_ranges=ranges)

    # ---- Task 2: market-basket analysis --------------------------------------
    orders_seg = orders.merge(diners[["diner_id", "segment"]], on="diner_id")
    baskets = pd.crosstab(orders["order_id"], orders["item"]).gt(0)
    rules_all = mine_rules(baskets, 0.03)
    pairs = {"All diners": top_pairs(rules_all, 8)}
    for seg, grp in orders_seg.groupby("segment"):
        seg_baskets = pd.crosstab(grp["order_id"], grp["item"]).gt(0)
        try:
            pairs[seg] = top_pairs(mine_rules(seg_baskets, 0.03), 6)
        except Exception:                                   # no frequent itemsets
            pairs[seg] = top_pairs(pd.DataFrame(), 6)
    item_share = pd.crosstab(orders_seg["item"], orders_seg["segment"], normalize="columns")[SEGMENTS]
    basket_size = orders_seg.groupby(["segment", "order_id"]).size().groupby("segment").mean().reindex(SEGMENTS)
    reco = {seg: grp["item"].value_counts().head(5).index.tolist() for seg, grp in orders_seg.groupby("segment")}
    A.update(orders_seg=orders_seg, n_baskets=len(baskets), pairs=pairs, item_share=item_share,
             basket_size=basket_size, reco=reco)

    # ---- Task 3: reorder classifier ------------------------------------------
    X, y = diners[CLF_FEATURES], diners["reordered"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    baseline = float(y_te.value_counts(normalize=True).max())
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)                    # rows actual [churned, reordered]
    report = classification_report(y_te, y_pred, target_names=["churned", "reordered"], output_dict=True)
    cv = cross_val_score(RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
                         X, y, cv=5)
    oof = cross_val_predict(RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
                            X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
    churned = (y == 0).to_numpy()
    thr_rows = []
    for t in np.arange(0.10, 0.91, 0.05):
        flagged = (1 - oof) >= t                           # flag as churn risk
        thr_rows.append({"threshold": round(float(t), 2),
                         "churners_caught": flagged[churned].mean(),
                         "loyal_flagged": flagged[~churned].mean(),
                         "precision": flagged[churned].sum() / max(flagged.sum(), 1)})
    A.update(clf=clf, clf_metrics={
        "accuracy": float(accuracy_score(y_te, y_pred)), "baseline": baseline, "cm": cm,
        "n_test": int(len(y_te)), "churn_caught": int(cm[0, 0]), "churn_missed": int(cm[0, 1]),
        "false_alarms": int(cm[1, 0]), "loyal_ok": int(cm[1, 1]), "churn_recall": float(report["churned"]["recall"]),
        "churn_precision": float(report["churned"]["precision"]), "cv_mean": float(cv.mean()),
        "cv_folds": cv.round(3).tolist()},
        clf_importance=pd.Series(clf.feature_importances_, index=CLF_FEATURES).sort_values(ascending=False),
        threshold_table=pd.DataFrame(thr_rows),
        oof_reorder=pd.Series(oof, index=diners.index, name="p_reorder"),
        reorder_corr=diners[CLF_FEATURES].corrwith(diners["reordered"]))

    # ---- Bonus: lifetime-spend regression ------------------------------------
    Xr, yr = diners[REG_FEATURES], diners["total_spend"]
    kf = KFold(5, shuffle=True, random_state=42)
    candidates = {"Linear regression": LinearRegression(),
                  "Random forest": RandomForestRegressor(n_estimators=300, random_state=42),
                  "Gradient boosting": GradientBoostingRegressor(random_state=42)}
    cv_r2 = {n: float(cross_val_score(m, Xr, yr, cv=kf, scoring="r2").mean()) for n, m in candidates.items()}
    best = max(cv_r2, key=cv_r2.get)
    oof_spend = cross_val_predict(candidates[best], Xr, yr, cv=kf)
    final = candidates[best].fit(Xr, yr)
    imp = getattr(final, "feature_importances_", None)
    if imp is None:
        imp = np.abs(final.coef_) / np.abs(final.coef_).sum()
    A.update(reg={
        "cv_r2": cv_r2, "best": best, "r2": float(r2_score(yr, oof_spend)),
        "mae": float(mean_absolute_error(yr, oof_spend)), "oof_pred": pd.Series(oof_spend, index=diners.index),
        "importance": pd.Series(imp, index=REG_FEATURES).sort_values(ascending=False),
        "corr": diners[REG_FEATURES].corrwith(diners["total_spend"])})
    return A


# ----------------------------------------------------------------------------
# Live simulator + assignment export
# ----------------------------------------------------------------------------
def analyse_diner(A: dict, profile: dict) -> dict:
    """Segment, reorder probability and recommendations for one diner profile."""
    row = pd.DataFrame([profile])
    cluster = int(A["kmeans"].predict(A["scaler"].transform(row[FEATURES]))[0])
    seg = A["segment_names"][cluster]
    p_reorder = float(A["clf"].predict_proba(row[CLF_FEATURES])[0][1])
    pairs = A["pairs"].get(seg, pd.DataFrame())
    strong = pairs[pairs["lift"] >= 1.5].head(2) if len(pairs) else pairs
    return {"segment": seg, "p_reorder": p_reorder, "items": A["reco"].get(seg, []),
            "bundles": strong.to_dict("records")}


def export_models(A: dict, path: Path | str = ROOT / "food_models.joblib") -> Path:
    """Save the models in the exact dictionary format the original case-study app.py expects."""
    joblib.dump({"scaler": A["scaler"], "kmeans": A["kmeans"], "segment_names": A["segment_names"],
                 "clf": A["clf"], "features": FEATURES, "reco": A["reco"]}, path)
    return Path(path)


# ----------------------------------------------------------------------------
# What-if tools for the simulator
# ----------------------------------------------------------------------------
def segment_distances(A: dict, profile: dict) -> pd.Series:
    """Distance (in scaled units) from a diner to each segment's centre. Lower = closer = better fit."""
    row = pd.DataFrame([profile])
    z = A["scaler"].transform(row[FEATURES])[0]
    centres = A["kmeans"].cluster_centers_
    dist = np.sqrt(((centres - z) ** 2).sum(axis=1))
    return pd.Series({A["segment_names"][i]: float(dist[i]) for i in range(len(dist))}).reindex(SEGMENTS)


def sweep_feature(A: dict, profile: dict, feature: str, n: int = 41) -> pd.DataFrame:
    """Vary one feature across its observed range, keep the rest fixed; return segment and reorder probability."""
    lo, hi = float(A["diners"][feature].min()), float(A["diners"][feature].max())
    values = np.linspace(lo, hi, n)
    rows = pd.DataFrame([{**profile, feature: v} for v in values])
    clusters = A["kmeans"].predict(A["scaler"].transform(rows[FEATURES]))
    return pd.DataFrame({"value": values,
                         "segment": [A["segment_names"][int(c)] for c in clusters],
                         "p_reorder": A["clf"].predict_proba(rows[CLF_FEATURES])[:, 1]})
