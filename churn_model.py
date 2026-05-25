"""
churn_model.py
Logistic regression churn prediction model with feature importance.
Run after simulate_data.py.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import os

print("=" * 55)
print("  CHURN PREDICTION MODEL")
print("=" * 55)

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv('data/raw/customers.csv')
print(f"\nLoaded {len(df):,} records  |  Churn rate: {df['churned'].mean():.1%}\n")

# ── Feature engineering ────────────────────────────────────
le_plan = LabelEncoder()
le_channel = LabelEncoder()
df['plan_enc'] = le_plan.fit_transform(df['plan'])
df['channel_enc'] = le_channel.fit_transform(df['acquisition_channel'])

FEATURES = [
    'plan_enc', 'channel_enc', 'mrr', 'tenure_days',
    'avg_weekly_sessions', 'usage_drop_50pct', 'days_since_last_login',
    'inactive_7_days', 'open_support_tickets', 'nps_score',
    'weeks_no_email_open', 'integration_removed',
    'feature_dashboard', 'feature_integrations', 'feature_reports',
    'feature_api', 'feature_sharing',
]

FEATURE_LABELS = [
    'Plan type', 'Acq. channel', 'MRR ($)', 'Tenure (days)',
    'Weekly sessions', 'Usage drop 50%+', 'Days since login',
    'Inactive 7+ days', 'Open tickets', 'NPS score',
    'Weeks no email open', 'Integration removed',
    'Uses: Dashboard', 'Uses: Integrations', 'Uses: Reports',
    'Uses: API', 'Uses: Sharing',
]

X = df[FEATURES]
y = df['churned']

# ── Train/test split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── Model ──────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

# ── Results ────────────────────────────────────────────────
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {auc:.3f}\n")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

# ── Feature importance ─────────────────────────────────────
coefs = pd.Series(model.coef_[0], index=FEATURE_LABELS)
coefs_sorted = coefs.abs().sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Churn Prediction Model — Feature Importance', fontsize=14, fontweight='bold', y=1.01)

# Feature importance
colors = ['#E24B4A' if coefs[f] > 0 else '#378ADD' for f in coefs_sorted.index]
axes[0].barh(coefs_sorted.index, coefs_sorted.values, color=colors, edgecolor='none')
axes[0].set_xlabel('Coefficient magnitude (absolute)', fontsize=11)
axes[0].set_title('Feature influence on churn probability', fontsize=12)
axes[0].tick_params(axis='both', labelsize=10)
axes[0].spines[['top','right']].set_visible(False)

# Churn probability distribution
retained_proba = y_proba[y_test == 0]
churned_proba = y_proba[y_test == 1]
axes[1].hist(retained_proba, bins=30, alpha=0.7, color='#378ADD', label='Retained', edgecolor='none')
axes[1].hist(churned_proba, bins=30, alpha=0.7, color='#E24B4A', label='Churned', edgecolor='none')
axes[1].set_xlabel('Predicted churn probability', fontsize=11)
axes[1].set_ylabel('Customer count', fontsize=11)
axes[1].set_title('Predicted probability distribution', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].spines[['top','right']].set_visible(False)

plt.tight_layout()
os.makedirs('data/processed', exist_ok=True)
plt.savefig('data/processed/model_output.png', dpi=150, bbox_inches='tight')
print("\n✅ Chart saved → data/processed/model_output.png")

# ── Score all customers ────────────────────────────────────
X_all_s = scaler.transform(X[FEATURES])
df['churn_probability'] = model.predict_proba(X_all_s)[:, 1]
df['risk_tier'] = pd.cut(df['churn_probability'],
                          bins=[0, 0.2, 0.5, 0.75, 1.0],
                          labels=['Low', 'Medium', 'High', 'Critical'])

df[['customer_id','plan','mrr','churn_probability','risk_tier']]\
    .to_csv('data/processed/churn_scores.csv', index=False)
print("✅ Scores saved → data/processed/churn_scores.csv")

print(f"\nRisk tier breakdown:")
print(df['risk_tier'].value_counts().to_string())
print("\nTop 10 highest-risk customers:")
print(df[df['churned']==0].nlargest(10, 'churn_probability')[
    ['customer_id','plan','mrr','churn_probability','risk_tier']
].to_string(index=False))
