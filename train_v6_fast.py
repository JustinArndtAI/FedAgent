#!/usr/bin/env python
"""
V6 FAST - Build on V4 with DistilBERT boost
Uses existing V4 models and adds DistilBERT layer
"""
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
import xgboost as xgb
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import time
import sys
import io
import warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("V6 FAST - DISTILBERT ENHANCED V4")
print("=" * 80)
print("\nBuilding on V4's 93% with DistilBERT semantic boost\n")

start_time = time.time()

# Load existing V4 models
print("[1/5] Loading V4 models...")
try:
    v4_align = joblib.load('v4_align_ensemble.pkl')
    v4_wellbeing = joblib.load('v4_wellbeing_ensemble.pkl')
    v4_align_vec = joblib.load('v4_align_vectorizer.pkl')
    v4_wellbeing_vec = joblib.load('v4_wellbeing_vectorizer.pkl')
    print("[OK] V4 models loaded successfully")
except Exception as e:
    print(f"[ERROR] V4 models not found: {e}")
    print("Please run train_v4_lite.py first")
    exit(1)

# Setup DistilBERT
print("\n[2/5] Setting up DistilBERT...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
distilbert.eval()
print("[OK] DistilBERT loaded")

def get_distilbert_embedding(text):
    """Fast single text embedding"""
    inputs = tokenizer([text], padding=True, truncation=True, 
                      max_length=128, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = distilbert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

# Load test data from V5
print("\n[3/5] Loading test data...")
test_texts = [
    # Professional alignment (should score high)
    "I understand you're going through a difficult time. Your feelings are completely valid.",
    "Thank you for trusting me with this. Let's explore what would be most helpful.",
    "I hear the pain in your words, and I want to acknowledge how hard this must be.",
    "Your wellbeing is my priority. Together, we can work through this.",
    "It takes courage to share these feelings. You're not alone.",
    
    # Poor alignment (should score low)
    "Just get over it already.",
    "That's not a real problem.",
    "Stop being so dramatic.",
    
    # Positive wellbeing
    "I'm feeling good today",
    "Things are looking up",
    "I'm optimistic about the future",
    
    # Negative wellbeing
    "I'm so depressed",
    "Everything feels hopeless",
    "I can't go on anymore"
]

# Create meta-learner training data
print("\n[4/5] Creating V6 meta-learner...")
print("Generating meta-features from V4 predictions + DistilBERT...")

meta_features_align = []
meta_features_wellbeing = []
meta_labels_align = []
meta_labels_wellbeing = []

# Generate synthetic training data for meta-learner
from data_gen_v3 import generate_training_data
print("Generating training samples...")
samples = generate_training_data(1000)  # Quick 1000 samples

for i, (text, align_label, wb_label) in enumerate(samples):
    if i % 100 == 0:
        print(f"  Processing sample {i}/1000...")
    
    # Get V4 predictions
    tfidf_align = v4_align_vec.transform([text])
    tfidf_wellbeing = v4_wellbeing_vec.transform([text])
    
    v4_align_pred = v4_align.predict_proba(tfidf_align)[0]
    v4_wb_pred = v4_wellbeing.predict_proba(tfidf_wellbeing)[0]
    
    # Get DistilBERT embedding (reduced to 50 dims via PCA simulation)
    db_emb = get_distilbert_embedding(text)
    db_emb_reduced = db_emb[:50]  # Simple dimension reduction
    
    # Combine features
    meta_feat_align = np.concatenate([v4_align_pred, db_emb_reduced])
    meta_feat_wellbeing = np.concatenate([v4_wb_pred, db_emb_reduced])
    
    meta_features_align.append(meta_feat_align)
    meta_features_wellbeing.append(meta_feat_wellbeing)
    meta_labels_align.append(align_label)
    meta_labels_wellbeing.append(wb_label)

X_align = np.array(meta_features_align)
y_align = np.array(meta_labels_align)
X_wellbeing = np.array(meta_features_wellbeing)
y_wellbeing = np.array(meta_labels_wellbeing)

print(f"Meta-feature shape: {X_align.shape}")

# Train meta-learners
print("\nTraining V6 meta-learners...")

# Alignment meta-learner
v6_align = VotingClassifier([
    ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='logloss')),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=5)),
    ('lr', LogisticRegression(max_iter=1000))
], voting='soft')
v6_align.fit(X_align, y_align)

# Wellbeing meta-learner
v6_wellbeing = VotingClassifier([
    ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='logloss')),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=5)),
    ('lr', LogisticRegression(max_iter=1000))
], voting='soft')
v6_wellbeing.fit(X_wellbeing, y_wellbeing)

print("[OK] Meta-learners trained")

# Test on specific phrases
print("\n[5/5] Testing V6 performance...")
print("-" * 60)

test_results = []
for text in test_texts:
    # Get features
    tfidf_align = v4_align_vec.transform([text])
    tfidf_wellbeing = v4_wellbeing_vec.transform([text])
    
    v4_align_pred = v4_align.predict_proba(tfidf_align)[0]
    v4_wb_pred = v4_wellbeing.predict_proba(tfidf_wellbeing)[0]
    
    db_emb = get_distilbert_embedding(text)
    db_emb_reduced = db_emb[:50]
    
    # Meta features
    meta_align = np.concatenate([v4_align_pred, db_emb_reduced]).reshape(1, -1)
    meta_wellbeing = np.concatenate([v4_wb_pred, db_emb_reduced]).reshape(1, -1)
    
    # V6 predictions
    align_score = v6_align.predict_proba(meta_align)[0][1] * 100
    wellbeing_score = v6_wellbeing.predict_proba(meta_wellbeing)[0][1] * 100
    
    test_results.append({
        'text': text[:60] + '...' if len(text) > 60 else text,
        'alignment': align_score,
        'wellbeing': wellbeing_score
    })
    
    print(f"Text: {text[:50]}...")
    print(f"  Alignment: {align_score:.1f}%")
    print(f"  Wellbeing: {wellbeing_score:.1f}%")
    print()

# Save models
print("Saving V6 models...")
joblib.dump(v6_align, 'v6_align_meta.pkl')
joblib.dump(v6_wellbeing, 'v6_wellbeing_meta.pkl')

# Save config
config = {
    "type": "meta_learner",
    "base_models": "V4",
    "enhancement": "DistilBERT_50dim",
    "meta_features": 52,
    "training_samples": 1000
}
with open('v6_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Generate report
elapsed = time.time() - start_time
report = {
    "version": "V6 FAST",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "training_time_seconds": elapsed,
    "architecture": "V4 + DistilBERT meta-learning",
    "test_results": test_results,
    "base_accuracy": {
        "v4_alignment": 95.6,
        "v4_wellbeing": 90.4
    }
}

with open('v6_fast_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("=" * 80)
print("V6 FAST TRAINING COMPLETE")
print("=" * 80)
print(f"Training time: {elapsed:.1f} seconds")
print("\nModel saved as v6_align_meta.pkl and v6_wellbeing_meta.pkl")
print("Run test_v6_final.py for full evaluation")
print("=" * 80)