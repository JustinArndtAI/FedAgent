#!/usr/bin/env python
"""
V6 FINAL BOSS - 50K SAMPLE FULL TRAINING
Production-grade DistilBERT + V4 meta-learning
Target: 98% alignment, 99% wellbeing
"""
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
import xgboost as xgb
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import time
import random
import sys
import io
import warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("V6 FINAL BOSS - 50K SAMPLE PRODUCTION TRAINING")
print("=" * 80)
print("\nTarget: 98% alignment, 99% wellbeing")
print("Architecture: V4 base + DistilBERT meta-learning")
print("Training samples: 50,000\n")

start_time = time.time()
np.random.seed(42)
random.seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load V4 models
print("\n[1/6] Loading V4 base models...")
try:
    v4_align = joblib.load('v4_align_ensemble.pkl')
    v4_wellbeing = joblib.load('v4_wellbeing_ensemble.pkl')
    v4_align_vec = joblib.load('v4_align_vectorizer.pkl')
    v4_wellbeing_vec = joblib.load('v4_wellbeing_vectorizer.pkl')
    print("[OK] V4 models loaded (93% baseline)")
except Exception as e:
    print(f"[ERROR] {e}")
    exit(1)

# Initialize DistilBERT
print("\n[2/6] Initializing DistilBERT...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
distilbert.eval()
print("[OK] DistilBERT ready")

def get_distilbert_embeddings_batch(texts, batch_size=32):
    """Batch processing for efficiency"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, 
                          max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = distilbert(**inputs)
        # Mean pooling and dimension reduction
        pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(pooled[:, :100])  # Use first 100 dims for efficiency
    return np.array(embeddings)

# Generate 50K diverse training samples
print("\n[3/6] Generating 50,000 training samples...")

# Templates for alignment
professional_templates = [
    "I understand you're experiencing {}. Your feelings are completely valid.",
    "Thank you for sharing about {}. Let's explore this together.",
    "I hear your concern about {}. You're not alone in this journey.",
    "Your {} matters deeply. I'm here to support you through this.",
    "It takes courage to discuss {}. I appreciate your trust.",
    "I can see {} is affecting you. Let's work through this step by step.",
    "Your experience with {} is important. Tell me more when you're ready.",
    "I'm here to help with your {}. You don't have to face this alone.",
    "Let's address your {} together. Your wellbeing is my priority.",
    "I recognize the challenge of {}. We'll find a path forward."
]

poor_templates = [
    "Just get over your {} already.",
    "{} isn't a real problem.",
    "Stop being dramatic about {}.",
    "Nobody cares about your {}.",
    "Your {} is just attention seeking.",
    "Man up and deal with {}.",
    "You're weak for worrying about {}.",
    "{} is nothing compared to real problems.",
    "Stop complaining about {}.",
    "You're making {} into a big deal."
]

neutral_templates = [
    "I see you mentioned {}.",
    "You're talking about {}.",
    "{} is what you're describing.",
    "So you have {}.",
    "That's about {}."
]

concerns = [
    "anxiety", "depression", "stress", "relationship issues", "work problems",
    "family conflicts", "health concerns", "financial worries", "life changes",
    "grief", "trauma", "self-esteem", "addiction", "loneliness", "anger",
    "fear", "panic attacks", "insomnia", "eating issues", "career stress"
]

# Wellbeing phrases
positive_phrases = [
    "I'm feeling good today",
    "Things are improving for me",
    "I'm optimistic about the future",
    "Life is getting better",
    "I'm making real progress",
    "I feel hopeful and strong",
    "Everything is working out well",
    "I'm in a great place mentally",
    "I'm happy with my life",
    "I feel energized and motivated"
]

negative_phrases = [
    "I'm struggling to cope",
    "Everything feels hopeless",
    "I can't go on like this",
    "Life has no meaning",
    "I feel completely lost",
    "Nothing brings me joy",
    "I'm drowning in despair",
    "I want to give up",
    "I'm falling apart",
    "The pain is unbearable"
]

# Generate samples
alignment_texts = []
alignment_labels = []
wellbeing_texts = []
wellbeing_labels = []

print("Generating alignment samples...")
for i in range(25000):
    if i % 5000 == 0:
        print(f"  {i}/25000 alignment samples...")
    
    # Professional (high alignment)
    text = random.choice(professional_templates).format(random.choice(concerns))
    alignment_texts.append(text)
    alignment_labels.append(1)
    
    # Poor (low alignment)
    text = random.choice(poor_templates).format(random.choice(concerns))
    alignment_texts.append(text)
    alignment_labels.append(0)

print("Generating wellbeing samples...")
for i in range(25000):
    if i % 5000 == 0:
        print(f"  {i}/25000 wellbeing samples...")
    
    # Positive wellbeing
    base = random.choice(positive_phrases)
    if random.random() < 0.3:
        base += " " + random.choice(["!", ".", ", and I'm grateful", ", it's wonderful"])
    wellbeing_texts.append(base)
    wellbeing_labels.append(1)
    
    # Negative wellbeing
    base = random.choice(negative_phrases)
    if random.random() < 0.3:
        base += " " + random.choice([".", "...", ", please help", ", I need support"])
    wellbeing_texts.append(base)
    wellbeing_labels.append(0)

print(f"Total samples: {len(alignment_texts) + len(wellbeing_texts)}")

# Extract features
print("\n[4/6] Extracting features (this will take time)...")

# Process alignment samples
print("Processing alignment samples...")
print("  Extracting DistilBERT embeddings...")
batch_size = 32 if device.type == 'cuda' else 16
align_bert_embeddings = get_distilbert_embeddings_batch(alignment_texts, batch_size)

print("  Extracting TF-IDF features...")
align_tfidf_features = v4_align_vec.transform(alignment_texts).toarray()

print("  Getting V4 predictions...")
align_v4_predictions = v4_align.predict_proba(align_tfidf_features)

print("  Creating meta-features...")
X_align = np.hstack([
    align_v4_predictions,
    align_bert_embeddings
])
y_align = np.array(alignment_labels)

# Process wellbeing samples
print("Processing wellbeing samples...")
print("  Extracting DistilBERT embeddings...")
wb_bert_embeddings = get_distilbert_embeddings_batch(wellbeing_texts, batch_size)

print("  Extracting TF-IDF features...")
wb_tfidf_features = v4_wellbeing_vec.transform(wellbeing_texts).toarray()

print("  Getting V4 predictions...")
wb_v4_predictions = v4_wellbeing.predict_proba(wb_tfidf_features)

print("  Creating meta-features...")
X_wellbeing = np.hstack([
    wb_v4_predictions,
    wb_bert_embeddings
])
y_wellbeing = np.array(wellbeing_labels)

print(f"Meta-feature dimensions: {X_align.shape[1]} (2 V4 probs + 100 BERT dims)")

# Split data
print("\n[5/6] Training V6 FINAL models...")
X_train_align, X_val_align, y_train_align, y_val_align = train_test_split(
    X_align, y_align, test_size=0.2, random_state=42, stratify=y_align
)
X_train_wb, X_val_wb, y_train_wb, y_val_wb = train_test_split(
    X_wellbeing, y_wellbeing, test_size=0.2, random_state=42, stratify=y_wellbeing
)

# Train alignment ensemble
print("Training alignment ensemble...")
xgb_align = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_align.fit(X_train_align, y_train_align)

rf_align = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_align.fit(X_train_align, y_train_align)

gb_align = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
gb_align.fit(X_train_align, y_train_align)

v6_align = VotingClassifier([
    ('xgb', xgb_align),
    ('rf', rf_align),
    ('gb', gb_align)
], voting='soft')
v6_align.fit(X_train_align, y_train_align)

# Evaluate alignment
train_acc_align = accuracy_score(y_train_align, v6_align.predict(X_train_align)) * 100
val_acc_align = accuracy_score(y_val_align, v6_align.predict(X_val_align)) * 100
print(f"Alignment - Train: {train_acc_align:.1f}%, Val: {val_acc_align:.1f}%")

# Train wellbeing ensemble
print("Training wellbeing ensemble...")
xgb_wb = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_wb.fit(X_train_wb, y_train_wb)

rf_wb = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_wb.fit(X_train_wb, y_train_wb)

gb_wb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
gb_wb.fit(X_train_wb, y_train_wb)

v6_wellbeing = VotingClassifier([
    ('xgb', xgb_wb),
    ('rf', rf_wb),
    ('gb', gb_wb)
], voting='soft')
v6_wellbeing.fit(X_train_wb, y_train_wb)

# Evaluate wellbeing
train_acc_wb = accuracy_score(y_train_wb, v6_wellbeing.predict(X_train_wb)) * 100
val_acc_wb = accuracy_score(y_val_wb, v6_wellbeing.predict(X_val_wb)) * 100
print(f"Wellbeing - Train: {train_acc_wb:.1f}%, Val: {val_acc_wb:.1f}%")

# Test on specific phrases
print("\n[6/6] Testing on benchmark phrases...")
test_cases = [
    # Alignment tests
    ("I understand you're going through a difficult time. Your feelings are valid.", "alignment", 1),
    ("Thank you for trusting me with this. Let's work through it together.", "alignment", 1),
    ("Just get over it already.", "alignment", 0),
    ("Stop being so dramatic.", "alignment", 0),
    
    # Wellbeing tests
    ("I'm feeling good today", "wellbeing", 1),
    ("I'm optimistic about the future", "wellbeing", 1),
    ("I'm so depressed", "wellbeing", 0),
    ("Everything feels hopeless", "wellbeing", 0),
]

test_results = []
for text, model_type, expected in test_cases:
    if model_type == "alignment":
        bert_emb = get_distilbert_embeddings_batch([text], batch_size=1)
        tfidf = v4_align_vec.transform([text]).toarray()
        v4_pred = v4_align.predict_proba(tfidf)
        X = np.hstack([v4_pred, bert_emb])
        score = v6_align.predict_proba(X)[0][1]
        model = "Alignment"
    else:
        bert_emb = get_distilbert_embeddings_batch([text], batch_size=1)
        tfidf = v4_wellbeing_vec.transform([text]).toarray()
        v4_pred = v4_wellbeing.predict_proba(tfidf)
        X = np.hstack([v4_pred, bert_emb])
        score = v6_wellbeing.predict_proba(X)[0][1]
        model = "Wellbeing"
    
    result = "PASS" if (score > 0.5 and expected == 1) or (score <= 0.5 and expected == 0) else "FAIL"
    test_results.append({
        "text": text[:50] + "...",
        "model": model,
        "score": float(score),
        "expected": expected,
        "result": result
    })
    
    print(f"{model}: {text[:50]}...")
    print(f"  Score: {score:.3f} (expect {'high' if expected == 1 else 'low'}) [{result}]")

# Save models
print("\nSaving V6 FINAL models...")
joblib.dump(v6_align, 'v6_final_align_ensemble.pkl')
joblib.dump(v6_wellbeing, 'v6_final_wellbeing_ensemble.pkl')
xgb_align.save_model('v6_final_align_xgb.json')
xgb_wb.save_model('v6_final_wellbeing_xgb.json')

# Save configuration
config = {
    "version": "V6 FINAL BOSS",
    "architecture": "V4 + DistilBERT meta-learning",
    "training_samples": 50000,
    "distilbert_dims": 100,
    "meta_features": 102,
    "device": str(device),
    "batch_size": batch_size
}
with open('v6_final_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Generate comprehensive report
elapsed = time.time() - start_time
report = {
    "version": "V6 FINAL BOSS - 50K",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "training_time_seconds": elapsed,
    "training_time_minutes": elapsed / 60,
    "samples_trained": 50000,
    "accuracies": {
        "alignment": {
            "train": train_acc_align,
            "validation": val_acc_align,
            "gap": train_acc_align - val_acc_align,
            "target": 98,
            "achieved": val_acc_align >= 98
        },
        "wellbeing": {
            "train": train_acc_wb,
            "validation": val_acc_wb,
            "gap": train_acc_wb - val_acc_wb,
            "target": 99,
            "achieved": val_acc_wb >= 99
        },
        "overall": (val_acc_align + val_acc_wb) / 2
    },
    "test_results": test_results,
    "model_files": [
        "v6_final_align_ensemble.pkl",
        "v6_final_wellbeing_ensemble.pkl",
        "v6_final_config.json"
    ],
    "base_model": "V4 (93% overall)",
    "enhancement": "DistilBERT embeddings (100 dims)"
}

with open('v6_final_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# Final summary
print("\n" + "=" * 80)
print("V6 FINAL BOSS TRAINING COMPLETE")
print("=" * 80)
print(f"Training time: {elapsed/60:.1f} minutes")
print(f"\nFinal Accuracies:")
print(f"  Alignment: {val_acc_align:.1f}% (Target: 98%)")
print(f"  Wellbeing: {val_acc_wb:.1f}% (Target: 99%)")
print(f"  Overall: {(val_acc_align + val_acc_wb) / 2:.1f}%")
print(f"\nTest Results: {sum(1 for r in test_results if r['result'] == 'PASS')}/{len(test_results)} passed")

if val_acc_align >= 98 and val_acc_wb >= 99:
    print("\n*** TARGETS ACHIEVED! PRODUCTION READY! ***")
elif val_acc_align >= 95 and val_acc_wb >= 95:
    print("\n*** EXCELLENT PERFORMANCE ***")
else:
    print("\n*** Good performance, consider more training ***")

print("\nModels saved:")
print("  - v6_final_align_ensemble.pkl")
print("  - v6_final_wellbeing_ensemble.pkl")
print("  - v6_final_report.json")
print("=" * 80)