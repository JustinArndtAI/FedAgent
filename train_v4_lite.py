#!/usr/bin/env python
"""
V4 LITE TRAINER - Faster training without DistilBERT
Focuses on advanced TF-IDF + XGBoost + Optuna for quick results
"""
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import sys
import io
import warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("🔥 V4 LITE TRAINER - FAST MODE 🔥")
print("=" * 80)
print("\nThis will train V4 models WITHOUT DistilBERT for faster completion")
print("Expected time: 3-5 minutes\n")

# ALIGNMENT MODEL
print("[1/2] Training V4 Alignment Model...")
print("-" * 40)

# Load data
texts = np.load("v4_align_texts.npy", allow_pickle=True)
labels = np.load("v4_align_labels.npy")
labels_binary = (labels >= 0.7).astype(int)

print(f"Loaded {len(texts)} alignment samples")

# TF-IDF with aggressive features
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 4),
    min_df=2,
    max_df=0.95,
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True
)

print("Extracting TF-IDF features...")
X = vectorizer.fit_transform(texts)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, labels_binary, test_size=0.2, random_state=42, stratify=labels_binary
)

# Train XGBoost with optimized params (simulated Optuna results)
print("Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# Train RandomForest
print("Training RandomForest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Create ensemble
print("Creating ensemble...")
ensemble = VotingClassifier(
    estimators=[('xgb', xgb_model), ('rf', rf_model)],
    voting='soft'
)
ensemble.fit(X_train, y_train)

# Evaluate
y_pred = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Alignment Accuracy: {accuracy*100:.1f}%")

# Save models
print("Saving alignment models...")
joblib.dump(ensemble, 'v4_align_ensemble.pkl')
joblib.dump(vectorizer, 'v4_align_vectorizer.pkl')
xgb_model.save_model('v4_align_xgb.json')
print("✓ Alignment models saved")

# WELLBEING MODEL
print("\n[2/2] Training V4 Wellbeing Model...")
print("-" * 40)

# Load data
with open("v4_wellbeing_texts.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f]
scores = np.load("v4_wellbeing_scores.npy")
labels_binary = (scores >= 0).astype(int)

print(f"Loaded {len(texts)} wellbeing samples")

# TF-IDF
vectorizer_wb = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 5),
    min_df=2,
    max_df=0.95,
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True
)

print("Extracting TF-IDF features...")
X = vectorizer_wb.fit_transform(texts)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, labels_binary, test_size=0.2, random_state=42, stratify=labels_binary
)

# Train models
print("Training XGBoost...")
xgb_wb = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_wb.fit(X_train, y_train)

print("Training RandomForest...")
rf_wb = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
rf_wb.fit(X_train, y_train)

# Create ensemble
print("Creating ensemble...")
ensemble_wb = VotingClassifier(
    estimators=[('xgb', xgb_wb), ('rf', rf_wb)],
    voting='soft'
)
ensemble_wb.fit(X_train, y_train)

# Evaluate
y_pred = ensemble_wb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Wellbeing Accuracy: {accuracy*100:.1f}%")

# Save models
print("Saving wellbeing models...")
joblib.dump(ensemble_wb, 'v4_wellbeing_ensemble.pkl')
joblib.dump(vectorizer_wb, 'v4_wellbeing_vectorizer.pkl')
xgb_wb.save_model('v4_wellbeing_xgb.json')
print("✓ Wellbeing models saved")

print("\n" + "=" * 80)
print("🔥 V4 LITE TRAINING COMPLETE! 🔥")
print("=" * 80)
print("\nModels saved:")
print("  - v4_align_ensemble.pkl")
print("  - v4_align_vectorizer.pkl")
print("  - v4_align_xgb.json")
print("  - v4_wellbeing_ensemble.pkl")
print("  - v4_wellbeing_vectorizer.pkl")
print("  - v4_wellbeing_xgb.json")
print("\nNow run: python run_v4_tests.py")
print("=" * 80)