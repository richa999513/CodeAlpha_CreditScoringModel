# ==========================================================
# COMPLETE CREDIT RISK TRAINING + SAVING SCRIPT
# ==========================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression

# ==========================================================
# 1️⃣ LOAD DATA
# ==========================================================

df = pd.read_csv("loan/credit_risk_dataset.csv")

print("Original Shape:", df.shape)

# ==========================================================
# 2️⃣ BASIC CLEANING
# ==========================================================

df = df.copy()

# Remove obvious useless columns if present
drop_cols = ['url', 'desc']  # adjust if needed
df = df.drop(columns=drop_cols, errors='ignore')

# Replace inf
df = df.replace([np.inf, -np.inf], np.nan)

# ==========================================================
# 3️⃣ TARGET CREATION
# ==========================================================

target_col = "loan_status"

# The provided dataset encodes loan status as 0/1. Handle numeric and
# textual labels robustly so this script works for both formats.
if pd.api.types.is_numeric_dtype(df[target_col]) or df[target_col].dropna().isin([0, 1]).all():
    y = df[target_col].astype(int)
else:
    bad_states = ['charged off', 'default', 'late', 'in grace period', 'collections']
    y = (df[target_col].astype(str).str.lower().isin(bad_states)).astype(int)

# Features + target
X = df.drop(columns=[target_col], errors="ignore")

# If a pre-computed target column exists, drop it from features
if "target_bin" in X.columns:
    X = X.drop(columns=["target_bin"])

print("Target Distribution:")
print(y.value_counts(normalize=True))

# ==========================================================
# 4️⃣ TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# 5️⃣ COLUMN TYPES
# ==========================================================

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Keep only low-cardinality categorical
low_card_cols = [c for c in cat_cols if X_train[c].nunique() <= 20]

print("Numeric columns:", len(numeric_cols))
print("Low-card categorical columns:", low_card_cols)

# ==========================================================
# 6️⃣ PREPROCESSOR
# ==========================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, low_card_cols)
])

# ==========================================================
# 7️⃣ FULL MODEL PIPELINE
# ==========================================================

model_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("variance", VarianceThreshold(0.0)),
    ("model", LogisticRegression(max_iter=1000))
])

# ==========================================================
# 8️⃣ TRAIN
# ==========================================================

model_pipeline.fit(X_train, y_train)

print("Train Score:", model_pipeline.score(X_train, y_train))
print("Test Score:", model_pipeline.score(X_test, y_test))

# ==========================================================
# 9️⃣ SAVE MODEL
# ==========================================================

joblib.dump(model_pipeline, "loan/credit_risk_model.pkl")

print("Model saved successfully as credit_risk_model.pkl")