# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json

from catboost import CatBoostClassifier

# ========== Load Best Results ==========
@st.cache_resource
def load_best_results():
    with open("best_results.json", "r") as f:
        results = json.load(f)
    return results

# ========== Load Model ==========

@st.cache_resource
def load_model():
    # Load best params for CatBoost from json
    results = load_best_results()
    cat_params = {}
    for k, v in results.get("CatBoost", {}).get("best_params", {}).items():
        cat_params[k.replace("classifier__", "")] = v

    # Load training data
    train_df = pd.read_csv("Train.csv")

    # Define columns
    drop_cols = ["ZONE1", "ZONE2","MRG","ARPU_SEGMENT","MONTANT","user_id"]
    target_col = "CHURN"
    cat_cols = ["REGION", "TOP_PACK", "TENURE"]

    X = train_df.drop(columns=[target_col] + drop_cols)
    y = train_df[target_col]

    # 🔑 Fix: handle NaN in categorical columns
    X[cat_cols] = X[cat_cols].fillna("missing").astype(str)

    # Initialize CatBoost with tuned params
    model = CatBoostClassifier(
        eval_metric="AUC",
        random_seed=42,
        verbose=0,
        **cat_params
    )

    # Fit while marking categorical features
    model.fit(X, y, cat_features=cat_cols)

    return model


# ========== Main App ==========
def main():
    st.set_page_config(page_title="Expresso Churn Prediction", layout="wide")
    st.title("📊 Expresso Churn Prediction App")
    st.write("Using tuned CatBoost model (parameters loaded from best_results.json).")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["EDA", "Prediction", "Model Info"])

    # ===== EDA Page =====
    if page == "EDA":
        st.header("🔍 Exploratory Data Analysis")

        df = pd.read_csv("Train.csv")
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Null Values (%)")
        null_percentage = (df.isnull().sum().sort_values(ascending=False) / len(df)) * 100
        st.bar_chart(null_percentage)

        st.subheader("CHURN Distribution")
        counts = df["CHURN"].value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        num_df = df.select_dtypes(include=['number'])
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(12,8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

    # ===== Prediction Page =====
    elif page == "Prediction":
        st.header("⚡ Predict Churn")

        test_df = pd.read_csv("Test.csv")
        st.write("Test Data Preview:", test_df.head())

        model = load_model()
        preds = model.predict(test_df.drop(columns=["user_id"], errors="ignore"))

        output = pd.DataFrame({
            "user_id": test_df["user_id"],
            "CHURN": preds
        })

        st.subheader("Prediction Results")
        st.dataframe(output)

        csv = output.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv, file_name="submission.csv", mime="text/csv")

    # ===== Model Info Page =====
    elif page == "Model Info":
        st.header("📘 Model Information")

        st.subheader("Best Results from Training")
        try:
            results = load_best_results()
            st.json(results)
        except:
            st.warning("No best_results.json found.")

        st.subheader("Training Log (CatBoost)")
        st.info("See catboost_training.json / learn_error.tsv / time_left.tsv for detailed logs.")

        st.success("Model reloaded with tuned parameters!")


if __name__ == "__main__":
    main()
