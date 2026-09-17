# Customer Churn Prediction — Dashboard

## Files
- `Customer_Churn_Prediction.ipynb` — full notebook: EDA, preprocessing, model training, evaluation
- `app.py` — Streamlit dashboard (Overview, EDA, Model Performance, live Prediction tool, Business Insights)
- `telco_churn.csv` — dataset (IBM Telco Customer Churn, 7,043 rows)
- `churn_model_xgboost.pkl`, `scaler.pkl`, `feature_columns.pkl` — trained model + preprocessing artifacts (produced by the notebook)
- `requirements.txt` — dependencies

## Setup

```bash
pip install -r requirements.txt
```

Make sure `telco_churn.csv`, `churn_model_xgboost.pkl`, `scaler.pkl`, and `feature_columns.pkl` are
in the same folder as `app.py` (run the notebook once first if the `.pkl` files aren't there yet —
the last cell of the notebook saves them).

## Run the dashboard

```bash
streamlit run app.py
```

This opens the dashboard in your browser at `http://localhost:8501`.

## Dashboard pages
1. **Overview** — headline KPIs (churn rate, avg tenure, avg charges) and top-level churn breakdowns
2. **Explore the Data** — filterable EDA (by contract, internet service, tenure range) with live charts
3. **Model Performance** — ROC curve, confusion matrix, feature importances, and an adjustable classification threshold slider
4. **Predict a Customer** — a form to enter a customer's profile and get a live churn risk score
5. **Business Insights** — key drivers and recommended retention actions
