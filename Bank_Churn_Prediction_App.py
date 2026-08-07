import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Customer Churn Predictor", page_icon="🏦", layout="wide")

# ---------------------------------------------------------
# Load data + train model (cached so it only runs once)
# ---------------------------------------------------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("Churn_Modelling.csv")

    df_model = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    df_model = pd.get_dummies(df_model, columns=['Geography', 'Gender'], drop_first=True)

    X = df_model.drop('Exited', axis=1)
    y = df_model['Exited']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    y_prob_test = xgb_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob_test)

    feat_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return xgb_model, X.columns.tolist(), roc_auc, feat_importance, X_test, y_test, y_prob_test


model, feature_columns, roc_auc, feat_importance, X_test, y_test, y_prob_test = load_and_train()

THRESHOLD = 0.3  # matches the notebook's tuned threshold (lowered from 0.5 to boost recall)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🏦 Bank Customer Churn Predictor")
st.markdown(
    "Predicts whether a bank customer is likely to churn, using an XGBoost model "
    "trained on 10,000 customer records. Adjust the inputs on the left and see a "
    "live prediction, or explore what drives churn in the **Model Insights** tab."
)

tab1, tab2 = st.tabs(["🔮 Predict", "📊 Model Insights"])

# ---------------------------------------------------------
# TAB 1 — Prediction
# ---------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.subheader("Customer Details")

        credit_score = st.slider("Credit Score", 300, 900, 650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 18, 100, 40)
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
        balance = st.number_input("Account Balance (€)", min_value=0.0, value=75000.0, step=1000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
        is_active = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)
        estimated_salary = st.number_input("Estimated Salary (€)", min_value=0.0, value=100000.0, step=1000.0)

    # Build input row matching the training feature set exactly
    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": estimated_salary,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
    }
    input_df = pd.DataFrame([input_dict])[feature_columns]

    prob = model.predict_proba(input_df)[0, 1]
    prediction = int(prob > THRESHOLD)

    with col2:
        st.subheader("Prediction")

        if prediction == 1:
            st.error(f"⚠️ **Likely to churn** — probability: {prob:.1%}")
        else:
            st.success(f"✅ **Likely to stay** — churn probability: {prob:.1%}")

        st.progress(min(float(prob), 1.0))
        st.caption(
            f"Decision threshold set at {THRESHOLD:.0%} (tuned down from the default 50% "
            "to catch more at-risk customers — missing a churner is costlier than a false alert)."
        )

        st.markdown("#### Retention suggestions")
        tips = []
        if is_active == "No":
            tips.append("Customer is **inactive** — re-engagement campaign recommended.")
        if num_products == 1:
            tips.append("Customer holds only **1 product** — cross-sell opportunity.")
        if geography == "Germany":
            tips.append("**Germany** customers show the highest churn rate in this dataset.")
        if age > 40:
            tips.append("Customers **over 40** show elevated churn risk — consider tailored offers.")
        if not tips:
            st.write("No major risk flags — customer profile looks stable.")
        else:
            for t in tips:
                st.write("- " + t)

# ---------------------------------------------------------
# TAB 2 — Model Insights
# ---------------------------------------------------------
with tab2:
    st.subheader("Model Performance")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "XGBoost")
    c2.metric("ROC-AUC", f"{roc_auc:.3f}")
    c3.metric("Decision Threshold", f"{THRESHOLD:.0%}")

    st.markdown("#### Feature Importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feat_importance['Feature'], feat_importance['Importance'], color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title("XGBoost Feature Importance")
    st.pyplot(fig)

    st.markdown("#### Classification Report (test set, threshold = 0.3)")
    y_pred_adjusted = (y_prob_test > THRESHOLD).astype(int)
    report = classification_report(y_test, y_pred_adjusted, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(2)
    st.dataframe(report_df, use_container_width=True)

    st.markdown("#### Key Business Recommendations")
    st.markdown(
        """
- Target **inactive customers** with re-engagement campaigns.
- Focus retention efforts on customers holding **only 1 product**.
- Prioritize churn prevention strategies in **Germany**.
- Develop tailored offerings for customers **above 40**.
- Use the adjusted probability threshold in retention campaigns to maximize churn capture.
        """
    )

st.divider()
st.caption("Built with XGBoost + Streamlit · Model trained on the Churn_Modelling dataset (10,000 records)")
