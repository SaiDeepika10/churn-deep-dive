"""
simulate_data.py
Generates realistic synthetic customer churn data for analysis.
Outputs: data/raw/customers.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

N = 12400
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 6, 30)

PLANS = ['free', 'starter', 'pro', 'enterprise']
PLAN_WEIGHTS = [0.45, 0.30, 0.18, 0.07]
PLAN_MRR = {'free': 0, 'starter': 29, 'pro': 99, 'enterprise': 499}

CHANNELS = ['paid_social', 'display_ads', 'email_promo', 'organic_search', 'referral']
CHANNEL_WEIGHTS = [0.28, 0.22, 0.20, 0.18, 0.12]

CHURN_REASONS = ['price', 'missing_features', 'poor_onboarding', 'bugs_reliability', 'other']
REASON_WEIGHTS = [0.34, 0.27, 0.19, 0.12, 0.08]

BASE_CHURN_RATES = {
    'free': 0.14, 'starter': 0.09, 'pro': 0.05, 'enterprise': 0.02
}
CHANNEL_CHURN_MULTIPLIER = {
    'paid_social': 1.8, 'display_ads': 1.45, 'email_promo': 1.0,
    'organic_search': 0.51, 'referral': 0.29
}

def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

def simulate_customer(i):
    plan = random.choices(PLANS, PLAN_WEIGHTS)[0]
    channel = random.choices(CHANNELS, CHANNEL_WEIGHTS)[0]
    signup_date = random_date(START_DATE, END_DATE - timedelta(days=30))
    tenure_days = (END_DATE - signup_date).days

    base_rate = BASE_CHURN_RATES[plan]
    churn_prob = min(base_rate * CHANNEL_CHURN_MULTIPLIER[channel], 0.95)

    # New customers churn more
    if tenure_days < 30:
        churn_prob *= 2.2
    elif tenure_days < 90:
        churn_prob *= 1.5

    churned = random.random() < churn_prob

    # Feature activation (correlated with churn)
    activation_boost = 0.7 if churned else 1.0
    features_activated = {
        'feature_dashboard': random.random() < (0.82 * activation_boost),
        'feature_integrations': random.random() < (0.58 * activation_boost),
        'feature_reports': random.random() < (0.71 * activation_boost),
        'feature_api': random.random() < (0.43 * activation_boost),
        'feature_sharing': random.random() < (0.66 * activation_boost),
    }

    # Usage metrics
    if churned:
        avg_weekly_sessions = max(0, np.random.poisson(1.2))
        usage_drop = random.random() < 0.68
    else:
        avg_weekly_sessions = max(1, np.random.poisson(4.5))
        usage_drop = random.random() < 0.12

    # Inactivity
    days_since_last_login = np.random.exponential(3) if not churned else np.random.exponential(11)
    days_since_last_login = min(int(days_since_last_login), tenure_days)

    # Support tickets
    open_tickets = np.random.poisson(2.1) if churned else np.random.poisson(0.4)
    open_tickets = min(open_tickets, 8)

    # NPS
    nps_score = int(np.clip(np.random.normal(3.5 if churned else 7.2, 2.0), 0, 10))

    # Email engagement
    weeks_no_email_open = np.random.poisson(5) if churned else np.random.poisson(1)

    # Integration removed
    integration_removed = random.random() < (0.35 if churned else 0.05)

    record = {
        'customer_id': f'CUST_{i:05d}',
        'signup_date': signup_date.strftime('%Y-%m-%d'),
        'plan': plan,
        'mrr': PLAN_MRR[plan],
        'acquisition_channel': channel,
        'tenure_days': tenure_days,
        'churned': int(churned),
        'churn_date': (signup_date + timedelta(days=random.randint(1, tenure_days))).strftime('%Y-%m-%d') if churned else None,
        'churn_reason': random.choices(CHURN_REASONS, REASON_WEIGHTS)[0] if churned else None,
        'avg_weekly_sessions': round(avg_weekly_sessions, 1),
        'usage_drop_50pct': int(usage_drop),
        'days_since_last_login': days_since_last_login,
        'inactive_7_days': int(days_since_last_login >= 7),
        'open_support_tickets': open_tickets,
        'nps_score': nps_score,
        'weeks_no_email_open': weeks_no_email_open,
        'integration_removed': int(integration_removed),
        **{k: int(v) for k, v in features_activated.items()},
    }
    return record

print("Simulating customer data...")
records = [simulate_customer(i) for i in range(N)]
df = pd.DataFrame(records)

os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/raw/customers.csv', index=False)

print(f"✅ Generated {N} customer records")
print(f"   Churn rate: {df['churned'].mean():.1%}")
print(f"   Churned accounts: {df['churned'].sum()}")
print(f"   Avg MRR (churned): ${df[df['churned']==1]['mrr'].mean():.0f}")
print(f"   Saved → data/raw/customers.csv")
