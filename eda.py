import streamlit as st
import plotly.express as px
import pandas as pd

def run_eda(df):
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    # FIX: Monthly_Spend must be numeric before calling .mean()
    if "Monthly_Spend" in df.columns:
        avg = pd.to_numeric(df["Monthly_Spend"], errors="coerce").mean()
        col3.metric("Avg Monthly Spend", f"₹{avg:,.0f}")

    st.subheader("🎯 Target Distribution")
    if "Interest_in_MoodCart" in df.columns:
        counts = df["Interest_in_MoodCart"].value_counts().reset_index()
        counts.columns = ["Interest", "Count"]
        st.plotly_chart(
            px.bar(counts, x="Interest", y="Count",
                   color="Interest",
                   color_discrete_map={"Yes": "#4CAF50", "No": "#F44336", "Maybe": "#FF9800"},
                   text="Count"),
            use_container_width=True
        )

    st.subheader("😶 Mood Distribution")
    if "Mood" in df.columns:
        st.plotly_chart(
            px.histogram(df, x="Mood", color="Mood",
                         color_discrete_sequence=px.colors.qualitative.Pastel),
            use_container_width=True
        )

    if "Income" in df.columns and "Monthly_Spend" in df.columns:
        st.subheader("💰 Income vs Monthly Spend")
        df2 = df.copy()
        df2["Monthly_Spend"] = pd.to_numeric(df2["Monthly_Spend"], errors="coerce")
        income_order = ["<20k", "20k-50k", "50k-1L", ">1L"]
        st.plotly_chart(
            px.box(df2, x="Income", y="Monthly_Spend",
                   category_orders={"Income": income_order},
                   color="Income"),
            use_container_width=True
        )

    if "Mood" in df.columns and "Monthly_Spend" in df.columns:
        st.subheader("🧠 Mood vs Avg Spend")
        df3 = df.copy()
        df3["Monthly_Spend"] = pd.to_numeric(df3["Monthly_Spend"], errors="coerce")
        mood_spend = df3.groupby("Mood")["Monthly_Spend"].mean().reset_index()
        mood_spend.columns = ["Mood", "Avg_Spend"]
        st.plotly_chart(
            px.bar(mood_spend, x="Mood", y="Avg_Spend", color="Mood",
                   text=mood_spend["Avg_Spend"].round(0),
                   color_discrete_sequence=px.colors.qualitative.Set2),
            use_container_width=True
        )

    if "Decision_Style" in df.columns:
        st.subheader("🧩 Decision Style")
        st.plotly_chart(
            px.pie(df, names="Decision_Style", hole=0.4,
                   color_discrete_sequence=px.colors.qualitative.Pastel),
            use_container_width=True
        )

    if "Age" in df.columns:
        st.subheader("👥 Age Group Distribution")
        age_order = ["Under 18", "18-24", "25-34", "35-44", "45+"]
        age_counts = df["Age"].value_counts().reindex(age_order).reset_index()
        age_counts.columns = ["Age", "Count"]
        st.plotly_chart(
            px.bar(age_counts, x="Age", y="Count", color="Age",
                   color_discrete_sequence=px.colors.qualitative.Bold),
            use_container_width=True
        )
