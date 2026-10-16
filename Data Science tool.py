"""
Task 4: Data Science Tool Mastery Project
Portfolio Project: End-to-End Customer Churn Analytics & Machine Learning Pipeline
Libraries Demonstrated: NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

# Configure visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 150


# ==============================================================================
# 1. NUMPY MASTERY: Synthetic Dataset Generation & Vectorized Operations
# ==============================================================================
def build_dataset_with_numpy(n_samples=3000):
    print("=" * 65)
    print("1. NUMPY: Generating Synthetic Customer Dataset & Array Math")
    print("=" * 65)
    np.random.seed(42)

    tenure_months = np.random.randint(1, 73, size=n_samples)
    monthly_charges = np.random.normal(loc=70.0, scale=25.0, size=n_samples)
    monthly_charges = np.clip(monthly_charges, 20.0, 150.0).round(2)

    support_calls = np.random.poisson(lam=2.2, size=n_samples)
    contract_types = np.random.choice(
        ["Month-to-Month", "One-Year", "Two-Year"],
        size=n_samples,
        p=[0.55, 0.25, 0.20],
    )

    # Vectorized calculation of Total Charges with random noise
    total_charges = (tenure_months * monthly_charges * np.random.uniform(0.95, 1.05, n_samples)).round(2)

    # Vectorized churn probability logit calculation
    contract_risk = np.where(contract_types == "Month-to-Month", 1.2, np.where(contract_types == "One-Year", 0.3, -0.6))
    logit = (
        -2.0
        + 0.45 * support_calls
        + 0.018 * monthly_charges
        - 0.035 * tenure_months
        + contract_risk
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "CustomerID": np.arange(10001, 10001 + n_samples),
        "TenureMonths": tenure_months,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SupportCalls": support_calls,
        "ContractType": contract_types,
        "Churn": churn,
    })

    # Inject 25 missing values to demonstrate Pandas imputation
    missing_idx = np.random.choice(df.index, 25, replace=False)
    df.loc[missing_idx, "MonthlyCharges"] = np.nan

    print(f"[+] Created dataset with shape: {df.shape}")
    print(f"[+] Overall Churn Rate: { (df['Churn'].mean() * 100):.2f}%")
    return df


# ==============================================================================
# 2. PANDAS MASTERY: Data Cleaning, Wrangling & Feature Engineering
# ==============================================================================
def preprocess_with_pandas(df):
    print("\n" + "=" * 65)
    print("2. PANDAS: Data Wrangling, Imputation & Feature Engineering")
    print("=" * 65)

    # Group-based median imputation for missing values
    df["MonthlyCharges"] = df["MonthlyCharges"].fillna(
        df.groupby("ContractType")["MonthlyCharges"].transform("median")
    )

    # Feature Engineering
    df["SupportCallIntensity"] = (df["SupportCalls"] / (df["TenureMonths"] + 1)).round(4)
    df["LogTotalCharges"] = np.log1p(df["TotalCharges"]).round(4)
    df["IsHighMonthlySpend"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    print("\n--- Aggregated Summary by Contract Type ---")
    summary = df.groupby("ContractType").agg(
        Customer_Count=("CustomerID", "count"),
        Avg_Monthly_Charges=("MonthlyCharges", "mean"),
        Avg_Support_Calls=("SupportCalls", "mean"),
        Churn_Rate_Pct=("Churn", lambda x: round(x.mean() * 100, 2)),
    )
    print(summary)

    # One-Hot Encoding for Machine Learning
    df_encoded = pd.get_dummies(df.drop(columns=["CustomerID"]), columns=["ContractType"], drop_first=True, dtype=int)
    print(f"\n[+] Encoded Feature Matrix Shape: {df_encoded.shape}")
    return df, df_encoded


# ==============================================================================
# 3. SCIKIT-LEARN MASTERY: Supervised Machine Learning & Evaluation
# ==============================================================================
def train_and_evaluate_models(df_encoded):
    print("\n" + "=" * 65)
    print("3. SCIKIT-LEARN: Model Training, Scaling & Evaluation")
    print("=" * 65)

    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model 1: Logistic Regression
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

    # Model 2: Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

    # Compile evaluation metrics
    results = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Logistic_Regression": [
            accuracy_score(y_test, y_pred_lr),
            precision_score(y_test, y_pred_lr),
            recall_score(y_test, y_pred_lr),
            f1_score(y_test, y_pred_lr),
            roc_auc_score(y_test, y_prob_lr),
        ],
        "Random_Forest": [
            accuracy_score(y_test, y_pred_rf),
            precision_score(y_test, y_pred_rf),
            recall_score(y_test, y_pred_rf),
            f1_score(y_test, y_pred_rf),
            roc_auc_score(y_test, y_prob_rf),
        ],
    })
    print("\n--- Model Comparison Table ---")
    print(results.round(4).to_string(index=False))

    print("\n--- Detailed Classification Report (Random Forest) ---")
    print(classification_report(y_test, y_pred_rf))

    return X, y_test, y_prob_lr, y_prob_rf, rf_clf


# ==============================================================================
# 4. MATPLOTLIB & SEABORN MASTERY: Portfolio Visualizations
# ==============================================================================
def generate_portfolio_charts(df_encoded, X, y_test, y_prob_lr, y_prob_rf, rf_clf, output_dir="portfolio_charts"):
    print("\n" + "=" * 65)
    print("4. MATPLOTLIB & SEABORN: Generating Visual Portfolio Pieces")
    print("=" * 65)
    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df_encoded.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Matrix (Customer Churn Dataset)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_correlation_heatmap.png"))
    plt.close()
    print("[+] Saved: 01_correlation_heatmap.png")

    # Chart 2: Random Forest Feature Importance
    plt.figure(figsize=(10, 6))
    importances = pd.Series(rf_clf.feature_importances_, index=X.columns).sort_values(ascending=True)
    importances.plot(kind="barh", color="teal")
    plt.title("Random Forest Feature Importance Rankings", fontsize=14, fontweight="bold")
    plt.xlabel("Relative Importance Score", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_feature_importance.png"))
    plt.close()
    print("[+] Saved: 02_feature_importance.png")

    # Chart 3: ROC-AUC Curve Comparison
    plt.figure(figsize=(9, 6))
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_lr):.3f})", linewidth=2)
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.3f})", linewidth=2.5, color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess (AUC = 0.500)")

    plt.title("ROC Curve Comparison — Customer Churn Models", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_roc_auc_curve.png"))
    plt.close()
    print("[+] Saved: 03_roc_auc_curve.png")


if __name__ == "__main__":
    raw_df = build_dataset_with_numpy()
    clean_df, encoded_df = preprocess_with_pandas(raw_df)
    X_features, y_true, prob_lr, prob_rf, rf_model = train_and_evaluate_models(encoded_df)
    generate_portfolio_charts(encoded_df, X_features, y_true, prob_lr, prob_rf, rf_model)
    print("\n[SUCCESS] Data Science Tool Mastery Pipeline Executed Completely!")