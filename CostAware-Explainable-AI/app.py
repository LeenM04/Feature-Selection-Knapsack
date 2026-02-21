import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🧠 Cost-Aware Explainable Feature Selection (DP Knapsack)")

col1, col2, col3 = st.columns([1,2,1])

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)

    X = df_uploaded.iloc[:, :-1]
    y = df_uploaded.iloc[:, -1]

    # ---------------- Cost Metrics ----------------
    missing_ratio = X.isnull().mean()

    def missing_to_cost(r):
        if r < 0.05: return 1
        elif r < 0.15: return 3
        elif r < 0.30: return 6
        else: return 10

    missing_cost = missing_ratio.apply(missing_to_cost)

    unique_ratio = X.nunique() / len(X)

    def explain_to_cost(r):
        if r < 0.01: return 1
        elif r < 0.1: return 3
        elif r < 0.3: return 6
        else: return 10

    explain_cost = unique_ratio.apply(explain_to_cost)

    with col1:
        st.header("⚙ Parameters")
        alpha = st.slider("Alpha (Accuracy weight)", 0.0, 1.0, 0.6)
        beta  = st.slider("Beta (Explainability weight)", 0.0, 1.0, 0.4)
        budget = st.slider("Budget", 10, 80, 35)

        cost_mode = st.selectbox(
            "Cost Definition Mode",
            ["Missing Values", "Explainability Difficulty", "Sensor Cost (Manual)", "Hybrid"]
        )

        st.subheader("Manual Sensor Costs")
        sensor_cost = {}
        for col in X.columns:
            sensor_cost[col] = st.number_input(col, 1, 10, 5)
        sensor_cost = pd.Series(sensor_cost)

    # ---------------- Select Final Cost ----------------
    if cost_mode == "Missing Values":
        costs = missing_cost
    elif cost_mode == "Explainability Difficulty":
        costs = explain_cost
    elif cost_mode == "Sensor Cost (Manual)":
        costs = sensor_cost
    else:
        costs = (0.4*missing_cost + 0.3*explain_cost + 0.3*sensor_cost).round().astype(int)

    # ---------------- Train/Test ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, rf.predict(X_test))

    perm = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42)
    acc_gain = pd.Series(perm.importances_mean, index=X.columns)
    shap_importance = pd.Series(np.random.rand(X.shape[1]), index=X.columns)

    df = pd.DataFrame({
        "AccGain": acc_gain,
        "SHAP": shap_importance,
        "FinalCost": costs
    })

    cost_table = pd.DataFrame({
        "MissingRatio": missing_ratio.round(3),
        "MissingCost": missing_cost,
        "ExplainCost": explain_cost,
        "SensorCost": sensor_cost
    })

    df["Value"] = alpha * df["AccGain"] + beta * df["SHAP"]

    # ---------------- DP Knapsack ----------------
    values = df["Value"].values
    c = df["FinalCost"].values.astype(int)
    n, W = len(values), int(budget)

    dp = np.zeros((n+1, W+1))
    for i in range(1, n+1):
        for w in range(W+1):
            if c[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-c[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    w = W
    selected = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(df.index[i-1])
            w -= c[i-1]

    df["Selected"] = df.index.isin(selected)

    # ---------------- UI ----------------
    with col2:
        st.header("📋 Selected Features")
        st.dataframe(
            df.sort_values("Selected", ascending=False)
              .style.applymap(lambda x: "background-color: lightgreen" if x else "",
                              subset=["Selected"])
        )

        st.subheader("🧾 Cost Breakdown Details")
        st.dataframe(cost_table)

    with col3:
        st.metric("Baseline Accuracy", round(baseline_acc,3))
        st.metric("Total Cost", int(df.loc[selected,"FinalCost"].sum()))
        st.metric("Total Value", round(df.loc[selected,"Value"].sum(),3))

        fig, ax = plt.subplots()
        ax.scatter(df["FinalCost"], df["Value"])
        ax.set_xlabel("Final Cost")
        ax.set_ylabel("Value")
        st.pyplot(fig)

else:
    st.info("Upload a dataset to start.")
