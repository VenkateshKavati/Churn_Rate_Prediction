"""
Customer Churn Prediction — Interactive Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              classification_report, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide", page_icon="📉")

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Data loading & prep (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_raw():
    df = pd.read_csv("telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df


@st.cache_data
def preprocess(df):
    data = df.drop("customerID", axis=1).copy()
    data["Churn"] = (data["Churn"] == "Yes").astype(int)

    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    le = LabelEncoder()
    for col in binary_cols:
        data[col] = le.fit_transform(data[col])

    multi_cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                       "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                       "Contract", "PaymentMethod"]
    data = pd.get_dummies(data, columns=multi_cat_cols, drop_first=True)

    data["AvgMonthlySpend"] = data["TotalCharges"] / (data["tenure"] + 1)
    data["TenureGroup"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 60, 72],
                                  labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"],
                                  include_lowest=True)
    data = pd.get_dummies(data, columns=["TenureGroup"], drop_first=True)
    return data


@st.cache_resource
def load_model_artifacts():
    model = joblib.load("churn_model_xgboost.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns


raw_df = load_raw()
data = preprocess(raw_df)
model, scaler, feature_columns = load_model_artifacts()

X = data.drop("Churn", axis=1)
y = data["Churn"]
X = X.reindex(columns=feature_columns, fill_value=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
test_proba = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📉 Churn Dashboard")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Explore the Data", "Model Performance", "Predict a Customer", "Business Insights"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Telco Customer Churn dataset · 7,043 customers")

# ---------------------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Customer Churn Prediction Dashboard")
    st.markdown("An interactive companion to the churn prediction notebook — explore the data, "
                "check model performance, and score new customers live.")

    churn_rate = raw_df["Churn"].value_counts(normalize=True)["Yes"] * 100
    avg_tenure = raw_df["tenure"].mean()
    avg_monthly = raw_df["MonthlyCharges"].mean()
    total_customers = len(raw_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churn Rate", f"{churn_rate:.1f}%")
    c3.metric("Avg. Tenure", f"{avg_tenure:.0f} months")
    c4.metric("Avg. Monthly Charges", f"${avg_monthly:.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(4, 4))
        raw_df["Churn"].value_counts().plot.pie(
            autopct="%1.1f%%", colors=["#4C72B0", "#DD8452"], ax=ax, ylabel=""
        )
        st.pyplot(fig)

    with col2:
        st.subheader("Churn by Contract Type")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=raw_df, x="Contract", hue="Churn",
                       palette=["#4C72B0", "#DD8452"], ax=ax)
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# PAGE: Explore the Data
# ---------------------------------------------------------------------------
elif page == "Explore the Data":
    st.title("Explore the Data")

    with st.expander("Raw data sample"):
        st.dataframe(raw_df.head(50))

    st.subheader("Filter customers")
    col1, col2, col3 = st.columns(3)
    contract_filter = col1.multiselect("Contract", raw_df["Contract"].unique(),
                                        default=list(raw_df["Contract"].unique()))
    internet_filter = col2.multiselect("Internet Service", raw_df["InternetService"].unique(),
                                        default=list(raw_df["InternetService"].unique()))
    tenure_range = col3.slider("Tenure (months)", 0, 72, (0, 72))

    filtered = raw_df[
        raw_df["Contract"].isin(contract_filter)
        & raw_df["InternetService"].isin(internet_filter)
        & raw_df["tenure"].between(*tenure_range)
    ]
    st.caption(f"{len(filtered):,} customers match this filter · "
               f"churn rate: {(filtered['Churn'] == 'Yes').mean() * 100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tenure Distribution by Churn")
        fig, ax = plt.subplots()
        sns.histplot(data=filtered, x="tenure", hue="Churn", bins=25, kde=True,
                     palette=["#4C72B0", "#DD8452"], ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Charges by Churn")
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered, x="Churn", y="MonthlyCharges",
                    palette=["#4C72B0", "#DD8452"], ax=ax)
        st.pyplot(fig)

    st.subheader("Churn rate by categorical feature")
    cat_col = st.selectbox(
        "Choose a feature",
        ["InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity",
         "SeniorCitizen", "Dependents", "Partner", "PaperlessBilling"],
    )
    rate_by_cat = filtered.groupby(cat_col)["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).sort_values(ascending=False)
    fig, ax = plt.subplots()
    rate_by_cat.plot.bar(color="#DD8452", ax=ax)
    ax.set_ylabel("Churn rate (%)")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# PAGE: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")
    st.caption("Model: XGBoost classifier, evaluated on a held-out 20% test set (1,409 customers)")

    auc = roc_auc_score(y_test, test_proba)
    threshold = st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.05,
                           help="Lower threshold = catch more churners, but more false alarms")
    preds = (test_proba >= threshold).astype(int)
    report = classification_report(y_test, preds, output_dict=True, target_names=["No Churn", "Churn"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{auc:.3f}")
    c2.metric("Precision (Churn)", f"{report['Churn']['precision']:.3f}")
    c3.metric("Recall (Churn)", f"{report['Churn']['recall']:.3f}")
    c4.metric("F1 (Churn)", f"{report['Churn']['f1-score']:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, test_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"XGBoost (AUC={auc:.3f})", color="#4C72B0")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader(f"Confusion Matrix (threshold = {threshold:.2f})")
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=ax)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        st.pyplot(fig)

    st.subheader("Top Feature Importances")
    importances = pd.Series(model.feature_importances_, index=feature_columns) \
        .sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# PAGE: Predict a Customer
# ---------------------------------------------------------------------------
elif page == "Predict a Customer":
    st.title("Predict Churn for a Single Customer")
    st.markdown("Enter a customer's details to get a live churn risk score from the trained model.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", ["Male", "Female"])
        senior = c1.selectbox("Senior Citizen", [0, 1])
        partner = c1.selectbox("Has Partner", ["Yes", "No"])
        dependents = c1.selectbox("Has Dependents", ["Yes", "No"])

        tenure = c2.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = c2.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
        contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = c2.selectbox("Payment Method",
                                ["Electronic check", "Mailed check",
                                 "Bank transfer (automatic)", "Credit card (automatic)"])

        internet = c3.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        tech_support = c3.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        online_security = c3.selectbox("Online Security", ["Yes", "No", "No internet service"])
        paperless = c3.selectbox("Paperless Billing", ["Yes", "No"])

        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        total_charges = monthly_charges * max(tenure, 1)
        raw_input = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges, "Churn": "No",
        }])

        binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        le = LabelEncoder()
        combined = pd.concat([raw_df.drop("customerID", axis=1), raw_input], ignore_index=True)
        for col in binary_cols:
            combined[col] = LabelEncoder().fit_transform(combined[col])

        multi_cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                           "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                           "Contract", "PaymentMethod"]
        combined = pd.get_dummies(combined, columns=multi_cat_cols, drop_first=True)
        combined["Churn"] = (combined["Churn"] == "Yes").astype(int) if combined["Churn"].dtype == object else combined["Churn"]
        combined["AvgMonthlySpend"] = combined["TotalCharges"] / (combined["tenure"] + 1)

        full_tenure = pd.concat([raw_df["tenure"], pd.Series([tenure])], ignore_index=True)
        combined["TenureGroup"] = pd.cut(full_tenure, bins=[0, 12, 24, 48, 60, 72],
                                          labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"],
                                          include_lowest=True)
        combined = pd.get_dummies(combined, columns=["TenureGroup"], drop_first=True)

        new_row = combined.iloc[[-1]].drop("Churn", axis=1)
        new_row = new_row.reindex(columns=feature_columns, fill_value=0)

        risk = model.predict_proba(new_row)[0, 1]

        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Churn Risk Score", f"{risk * 100:.1f}%")
            if risk >= 0.5:
                st.error("⚠️ High risk of churn")
            elif risk >= 0.3:
                st.warning("⚡ Moderate risk")
            else:
                st.success("✅ Low risk")
        with c2:
            fig, ax = plt.subplots(figsize=(6, 1.2))
            ax.barh([0], [1], color="#e0e0e0")
            ax.barh([0], [risk], color="#DD8452" if risk >= 0.5 else "#4C72B0")
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Churn probability")
            st.pyplot(fig)

# ---------------------------------------------------------------------------
# PAGE: Business Insights
# ---------------------------------------------------------------------------
elif page == "Business Insights":
    st.title("Business Insights & Recommendations")

    st.markdown("""
    ### Key Drivers of Churn
    1. **Contract type** — month-to-month customers churn far more than those on 1–2 year contracts.
    2. **Tenure** — churn is concentrated in the first year; this is an onboarding problem as much as a pricing one.
    3. **Add-on services** — customers without tech support or online security churn more, suggesting they feel less supported.
    4. **Payment method / fiber internet** — electronic check + fiber optic customers show elevated churn, often tied to price sensitivity.

    ### Recommended Actions
    - **Incentivize contract upgrades** for month-to-month customers (small discount for 1-year lock-in).
    - **Build a 90-day onboarding program** — proactive check-ins for new customers, since churn risk is highest early.
    - **Offer trial bundles** of tech support / online security to high-risk segments.
    - **Operationalize this model**: score the active customer base monthly, route the top-risk decile to a retention team.

    ### How to measure success
    Track retention-rate lift for the top-risk decile against a holdout control group that receives no intervention —
    this isolates the causal effect of the retention program rather than just correlating with the score.
    """)

    st.info("Use the **Predict a Customer** page to test how these levers (contract, tenure, add-ons) "
            "move an individual customer's risk score in real time.")
