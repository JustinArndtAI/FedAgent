#!/usr/bin/env python
"""
FINAL BOSS V6 - DISTILBERT + TF-IDF HYBRID PRODUCTION MODEL
Target: 98% Alignment, 99% Wellbeing
"""
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import pickle
import joblib
import xgboost as xgb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import ks_2samp
import json
import time
import warnings
import sys
import io
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("FINAL BOSS V6 - DISTILBERT HYBRID PRODUCTION MODEL")
print("=" * 80)
print("\nTarget: 98% Alignment, 99% Wellbeing via DistilBERT + TF-IDF fusion")
print("Building on V4 Lite's 93% foundation\n")

start_time = time.time()

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load DistilBERT
print("\n[1/7] Loading DistilBERT model...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
distilbert.eval()
print("[OK] DistilBERT loaded successfully")

# Load V4 data
print("\n[2/7] Loading V4 training data...")
try:
    # Load alignment data
    align_texts = np.load("v4_align_texts.npy", allow_pickle=True)
    align_labels = np.load("v4_align_labels.npy")
    align_labels = (align_labels >= 0.7).astype(int)
    
    # Load wellbeing data  
    with open("v4_wellbeing_texts.txt", "r", encoding="utf-8") as f:
        wellbeing_texts = [line.strip() for line in f]
    wellbeing_scores = np.load("v4_wellbeing_scores.npy")
    wellbeing_labels = (wellbeing_scores >= 0).astype(int)
    
    print(f"[OK] Loaded {len(align_texts)} alignment samples")
    print(f"[OK] Loaded {len(wellbeing_texts)} wellbeing samples")
except:
    print("[WARNING] V4 data not found, generating synthetic data...")
    # Generate synthetic data if V4 not available
    from data_gen_v4 import generate_alignment_data, generate_wellbeing_data
    align_texts, align_labels = generate_alignment_data(50000)
    wellbeing_texts, wellbeing_labels = generate_wellbeing_data(50000)
    align_labels = (align_labels >= 0.7).astype(int)
    wellbeing_labels = (wellbeing_labels >= 0).astype(int)

# Split data
X_train_align, X_val_align, y_train_align, y_val_align = train_test_split(
    align_texts, align_labels, test_size=0.2, random_state=42, stratify=align_labels
)
X_train_wb, X_val_wb, y_train_wb, y_val_wb = train_test_split(
    wellbeing_texts, wellbeing_labels, test_size=0.2, random_state=42, stratify=wellbeing_labels
)

print(f"Training split: {len(X_train_align)} train, {len(X_val_align)} validation")

# DistilBERT embedding function
def get_distilbert_embeddings(texts, batch_size=16):
    """Extract DistilBERT embeddings with mean pooling"""
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        if i % (batch_size * 10) == 0:
            print(f"  Processing batch {i//batch_size + 1}/{total_batches}...")
        
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, 
                          max_length=128, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = distilbert(**inputs)
        
        # Mean pooling over sequence length
        pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(pooled)
    
    return np.vstack(embeddings)

# Extract embeddings for alignment
print("\n[3/7] Extracting DistilBERT embeddings for alignment...")
print("This will take time but worth it for 98%+ accuracy...")

# Sample subset for faster training if needed
SAMPLE_SIZE = min(10000, len(X_train_align))
sample_idx = np.random.choice(len(X_train_align), SAMPLE_SIZE, replace=False)
X_train_align_sample = [X_train_align[i] for i in sample_idx]
y_train_align_sample = y_train_align[sample_idx]

print(f"Using {SAMPLE_SIZE} samples for training")
db_train_align = get_distilbert_embeddings(X_train_align_sample, batch_size=32 if device.type=='cuda' else 8)
db_val_align = get_distilbert_embeddings(X_val_align[:2000], batch_size=32 if device.type=='cuda' else 8)

# Load V4 TF-IDF vectorizers
print("\n[4/7] Creating hybrid features (DistilBERT + TF-IDF)...")
try:
    tfidf_align = joblib.load('v4_align_vectorizer.pkl')
    print("[OK] Loaded existing V4 TF-IDF vectorizer")
except:
    print("Creating new TF-IDF vectorizer...")
    tfidf_align = TfidfVectorizer(max_features=10000, ngram_range=(1,4), min_df=2)
    tfidf_align.fit(X_train_align)

# Get TF-IDF features
tfidf_train_align = tfidf_align.transform(X_train_align_sample).toarray()
tfidf_val_align = tfidf_align.transform(X_val_align[:2000]).toarray()

# Create hybrid features
print("Concatenating DistilBERT (768 dims) + TF-IDF (10K dims)...")
X_train_hybrid_align = np.hstack((db_train_align, tfidf_train_align))
X_val_hybrid_align = np.hstack((db_val_align, tfidf_val_align))
y_val_align_sample = y_val_align[:2000]

print(f"Hybrid feature shape: {X_train_hybrid_align.shape}")

# Train alignment models
print("\n[5/7] Training V6 alignment ensemble...")

# XGBoost with tuned hyperparameters
print("Training XGBoost...")
xgb_align = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_align.fit(X_train_hybrid_align, y_train_align_sample)

# RandomForest
print("Training RandomForest...")
rf_align = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_align.fit(X_train_hybrid_align, y_train_align_sample)

# GradientBoosting
print("Training GradientBoosting...")
gb_align = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    random_state=42
)
gb_align.fit(X_train_hybrid_align, y_train_align_sample)

# Create ensemble
print("Creating voting ensemble...")
ensemble_align = VotingClassifier(
    estimators=[
        ('xgb', xgb_align),
        ('rf', rf_align),
        ('gb', gb_align)
    ],
    voting='soft'
)
ensemble_align.fit(X_train_hybrid_align, y_train_align_sample)

# Evaluate alignment
train_acc_align = accuracy_score(y_train_align_sample, ensemble_align.predict(X_train_hybrid_align)) * 100
val_acc_align = accuracy_score(y_val_align_sample, ensemble_align.predict(X_val_hybrid_align)) * 100
gap_align = train_acc_align - val_acc_align

print(f"\n[OK] V6 Alignment Results:")
print(f"   Training Accuracy: {train_acc_align:.1f}%")
print(f"   Validation Accuracy: {val_acc_align:.1f}%")
print(f"   Overfitting Gap: {gap_align:.1f}%")

# Train wellbeing models (similar process)
print("\n[6/7] Training V6 wellbeing ensemble...")

# Sample for wellbeing
SAMPLE_SIZE_WB = min(10000, len(X_train_wb))
sample_idx_wb = np.random.choice(len(X_train_wb), SAMPLE_SIZE_WB, replace=False)
X_train_wb_sample = [X_train_wb[i] for i in sample_idx_wb]
y_train_wb_sample = y_train_wb[sample_idx_wb]

print("Extracting DistilBERT embeddings for wellbeing...")
db_train_wb = get_distilbert_embeddings(X_train_wb_sample, batch_size=32 if device.type=='cuda' else 8)
db_val_wb = get_distilbert_embeddings(X_val_wb[:2000], batch_size=32 if device.type=='cuda' else 8)

# TF-IDF for wellbeing
try:
    tfidf_wb = joblib.load('v4_wellbeing_vectorizer.pkl')
except:
    tfidf_wb = TfidfVectorizer(max_features=15000, ngram_range=(1,4), min_df=2)
    tfidf_wb.fit(X_train_wb)

tfidf_train_wb = tfidf_wb.transform(X_train_wb_sample).toarray()
tfidf_val_wb = tfidf_wb.transform(X_val_wb[:2000]).toarray()

# Hybrid features
X_train_hybrid_wb = np.hstack((db_train_wb, tfidf_train_wb))
X_val_hybrid_wb = np.hstack((db_val_wb, tfidf_val_wb))
y_val_wb_sample = y_val_wb[:2000]

# Train wellbeing models
xgb_wb = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_wb.fit(X_train_hybrid_wb, y_train_wb_sample)

rf_wb = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_wb.fit(X_train_hybrid_wb, y_train_wb_sample)

gb_wb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    random_state=42
)
gb_wb.fit(X_train_hybrid_wb, y_train_wb_sample)

ensemble_wb = VotingClassifier(
    estimators=[
        ('xgb', xgb_wb),
        ('rf', rf_wb),
        ('gb', gb_wb)
    ],
    voting='soft'
)
ensemble_wb.fit(X_train_hybrid_wb, y_train_wb_sample)

# Evaluate wellbeing
train_acc_wb = accuracy_score(y_train_wb_sample, ensemble_wb.predict(X_train_hybrid_wb)) * 100
val_acc_wb = accuracy_score(y_val_wb_sample, ensemble_wb.predict(X_val_hybrid_wb)) * 100
gap_wb = train_acc_wb - val_acc_wb

print(f"\n[OK] V6 Wellbeing Results:")
print(f"   Training Accuracy: {train_acc_wb:.1f}%")
print(f"   Validation Accuracy: {val_acc_wb:.1f}%")
print(f"   Overfitting Gap: {gap_wb:.1f}%")

# Validation tests
print("\n[7/7] Running validation suite...")

# Adversarial validation
print("Running adversarial validation...")
combined = np.vstack((X_train_hybrid_align[:5000], X_val_hybrid_align[:1000]))
labels = np.hstack((np.zeros(5000), np.ones(1000)))
X_adv_train, X_adv_val, y_adv_train, y_adv_val = train_test_split(
    combined, labels, test_size=0.2, random_state=42
)
adv_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
adv_model.fit(X_adv_train, y_adv_train)
adv_preds = adv_model.predict_proba(X_adv_val)[:, 1]
auc = roc_auc_score(y_adv_val, adv_preds)
print(f"Adversarial AUC: {auc:.3f} (< 0.7 is good)")

# KS tests on sample features
print("Running KS tests...")
p_values = []
for i in range(min(100, X_train_hybrid_align.shape[1])):
    stat, p = ks_2samp(X_train_hybrid_align[:, i], X_val_hybrid_align[:, i])
    p_values.append(p)
shift_count = sum(p < 0.05 for p in p_values)
print(f"Features with distribution shift: {shift_count}/100")

# Save models
print("\n" + "=" * 60)
print("Saving V6 models...")
joblib.dump(ensemble_align, 'v6_align_ensemble.pkl')
joblib.dump(ensemble_wb, 'v6_wellbeing_ensemble.pkl')
joblib.dump(tfidf_align, 'v6_align_vectorizer.pkl')
joblib.dump(tfidf_wb, 'v6_wellbeing_vectorizer.pkl')
xgb_align.save_model('v6_align_xgb.json')
xgb_wb.save_model('v6_wellbeing_xgb.json')

# Save model config for production
config = {
    "distilbert_model": "distilbert-base-uncased",
    "tfidf_features_align": 10000,
    "tfidf_features_wellbeing": 15000,
    "embedding_dim": 768,
    "sample_size_training": SAMPLE_SIZE,
    "device": str(device)
}
with open("v6_model_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Generate report
elapsed = time.time() - start_time
report = {
    "version": "V6 FINAL BOSS",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "training_time_seconds": elapsed,
    "accuracies": {
        "alignment": {
            "train": train_acc_align,
            "validation": val_acc_align,
            "gap": gap_align,
            "target": 98,
            "achieved": val_acc_align >= 98
        },
        "wellbeing": {
            "train": train_acc_wb,
            "validation": val_acc_wb,
            "gap": gap_wb,
            "target": 99,
            "achieved": val_acc_wb >= 99
        },
        "overall": (val_acc_align + val_acc_wb) / 2
    },
    "validation": {
        "adversarial_auc": auc,
        "auc_pass": auc < 0.7,
        "ks_shift_features": shift_count,
        "overfitting_check": {
            "alignment": "PASS" if gap_align < 10 else "FAIL",
            "wellbeing": "PASS" if gap_wb < 10 else "FAIL"
        }
    },
    "models": {
        "architecture": "DistilBERT + TF-IDF Hybrid",
        "ensemble": "XGBoost + RandomForest + GradientBoosting",
        "feature_dims": X_train_hybrid_align.shape[1]
    }
}

with open("v6_training_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 80)
print("V6 FINAL BOSS TRAINING COMPLETE")
print("=" * 80)
print(f"Alignment: {val_acc_align:.1f}% (Target: 98%)")
print(f"Wellbeing: {val_acc_wb:.1f}% (Target: 99%)")
print(f"Overall: {(val_acc_align + val_acc_wb) / 2:.1f}%")
print(f"Training Time: {elapsed/60:.1f} minutes")

if val_acc_align >= 98 and val_acc_wb >= 99:
    print("\n*** TARGETS ACHIEVED! PRODUCTION READY! ***")
elif val_acc_align >= 95 and val_acc_wb >= 95:
    print("\n*** EXCELLENT PERFORMANCE - Close to targets! ***")
else:
    print("\n*** Good performance, consider hyperparameter tuning ***")

print("\nNext: Run test_v6_final.py for comprehensive testing")
print("=" * 80)