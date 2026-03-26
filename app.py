import streamlit as st
import pandas as pd

# ── These imports work ONLY when the files are named exactly:
#    utils.py  |  eda.py  |  models.py  |  app.py
# ── on Streamlit Cloud every file in your repo must use these exact names.
from utils import load_data
from eda import run_eda
from models import (
    train_classification, train_regression,
    train_clustering, association_mining,
    save_model, load_model, predict_new,
)

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MoodCart Analytics", layout="wide", page_icon="🛒")
st.title("🛒 MoodCart Analytics Dashboard")

menu = st.sidebar.selectbox(
    "Navigation",
    ["Upload Data", "EDA", "Classification", "Regression",
     "Clustering", "Association Rules", "Predict New"],
)

# FIX: initialise session state keys once
if "df" not in st.session_state:
    st.session_state["df"] = None

# ─────────────────────────────────────────────────────────────────────────────
# Upload Data
# ─────────────────────────────────────────────────────────────────────────────
if menu == "Upload Data":
    st.header("📂 Upload Dataset")
    f = st.file_uploader("Upload CSV", type=["csv"])
    if f:
        # FIX: cache the loaded df in session_state so it persists across pages
        st.session_state["df"] = load_data(f)
        st.success(f"✅ Data loaded — {len(st.session_state['df']):,} rows, "
                   f"{st.session_state['df'].shape[1]} columns")
        st.dataframe(st.session_state["df"].head(), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# EDA
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "EDA":
    if st.session_state["df"] is not None:
        run_eda(st.session_state["df"])
    else:
        st.warning("⚠️ Please upload data first.")

# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "Classification":
    if st.session_state["df"] is not None:
        st.header("🤖 Classification — Predict Interest in MoodCart")
        # FIX: wrap in button so training doesn't re-run on every widget interaction
        if st.button("▶ Train All Models", type="primary"):
            with st.spinner("Training 4 models… this may take ~30 seconds."):
                try:
                    res, best_model, le, cols = train_classification(st.session_state["df"])
                    save_model(best_model, le, cols)
                    st.session_state["clf_results"] = res
                    st.success(f"✅ Best model saved. "
                               f"Best F1: {res['F1 Score'].max():.4f}")
                except Exception as e:
                    st.error(f"Training error: {e}")

        if "clf_results" in st.session_state:
            st.subheader("Model Comparison")
            st.dataframe(
                st.session_state["clf_results"]
                  .style.highlight_max(subset=["Accuracy","F1 Score"], color="#d4edda"),
                use_container_width=True,
            )
    else:
        st.warning("⚠️ Please upload data first.")

# ─────────────────────────────────────────────────────────────────────────────
# Regression
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "Regression":
    if st.session_state["df"] is not None:
        st.header("📈 Regression — Predict Monthly Spend")
        if st.button("▶ Train Regression Models", type="primary"):
            with st.spinner("Training…"):
                try:
                    scores, _ = train_regression(st.session_state["df"])
                    # FIX: display as a proper table, not a raw dict
                    scores_df = pd.DataFrame(
                        list(scores.items()), columns=["Model", "R² Score"]
                    )
                    st.subheader("R² Scores")
                    st.dataframe(scores_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Training error: {e}")
    else:
        st.warning("⚠️ Please upload data first.")

# ─────────────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "Clustering":
    if st.session_state["df"] is not None:
        st.header("🔵 Clustering — KMeans Segments")
        k = st.slider("Number of clusters (k)", 2, 8, 4)
        if st.button("▶ Run Clustering", type="primary"):
            with st.spinner("Clustering…"):
                try:
                    labels = train_clustering(st.session_state["df"], k=k)
                    counts = pd.Series(labels, name="Cluster").value_counts().sort_index()
                    counts.index = [f"Cluster {i}" for i in counts.index]
                    st.subheader("Cluster Sizes")
                    st.bar_chart(counts)
                    st.dataframe(counts.reset_index().rename(
                        columns={"index": "Cluster", "Cluster": "Count"}),
                        use_container_width=True)
                except Exception as e:
                    st.error(f"Clustering error: {e}")
    else:
        st.warning("⚠️ Please upload data first.")

# ─────────────────────────────────────────────────────────────────────────────
# Association Rules
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "Association Rules":
    if st.session_state["df"] is not None:
        st.header("🔗 Association Rules — Product Combinations")
        col1, col2 = st.columns(2)
        min_sup  = col1.slider("Min Support",    0.01, 0.3, 0.05, 0.01)
        min_conf = col2.slider("Min Confidence", 0.1,  0.9, 0.3,  0.05)

        if st.button("▶ Mine Rules", type="primary"):
            with st.spinner("Running Apriori…"):
                try:
                    rules = association_mining(st.session_state["df"],
                                               min_support=min_sup)
                    if rules.empty:
                        st.warning("No rules found. Try lowering support/confidence.")
                    else:
                        # FIX: filter by confidence here since we expose it in UI
                        rules = rules[rules["confidence"] >= min_conf]
                        st.success(f"Found **{len(rules)}** rules.")
                        st.dataframe(
                            rules.head(30).style.format(
                                {"support": "{:.3f}", "confidence": "{:.3f}", "lift": "{:.2f}"}
                            ),
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"Association mining error: {e}")
    else:
        st.warning("⚠️ Please upload data first.")

# ─────────────────────────────────────────────────────────────────────────────
# Predict New
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "Predict New":
    st.header("🔮 Predict on New Data")
    f = st.file_uploader("Upload new CSV (same structure)", type=["csv"])
    if f:
        try:
            df_new = pd.read_csv(f)
            model, le, cols = load_model()
            preds = predict_new(df_new, model, le, cols)
            df_new["Prediction"] = preds
            st.success(f"✅ Predicted {len(df_new):,} rows.")
            st.dataframe(df_new[["Prediction"] + [c for c in df_new.columns
                                                   if c != "Prediction"]].head(30),
                         use_container_width=True)
            # FIX: offer download of predictions
            csv = df_new.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions CSV", csv,
                               "predictions.csv", "text/csv")
        except FileNotFoundError:
            st.error("❌ No trained model found. Please run Classification first.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("MoodCart Analytics · Streamlit Cloud")
