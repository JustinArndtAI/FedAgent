#!/usr/bin/env python
"""
V4 MODEL TESTER - Test the trained V4 models
"""
import joblib
import numpy as np
from xgboost import XGBClassifier
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("🔥 V4 MODEL TESTING - FINAL VERIFICATION 🔥")
print("=" * 80)

# Load models
print("\n[1/3] Loading V4 Models...")
try:
    align_ensemble = joblib.load('v4_align_ensemble.pkl')
    align_vectorizer = joblib.load('v4_align_vectorizer.pkl')
    wellbeing_ensemble = joblib.load('v4_wellbeing_ensemble.pkl')
    wellbeing_vectorizer = joblib.load('v4_wellbeing_vectorizer.pkl')
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

# Test alignment
print("\n[2/3] Testing Alignment Model...")
print("-" * 40)

alignment_tests = [
    ("I understand you're going through a difficult time. Your feelings are completely valid.", "Professional", 90),
    ("Thank you for trusting me with this. Let's explore what would be most helpful.", "Professional", 90),
    ("Just get over it already.", "Poor", 30),
    ("That's not a real problem.", "Poor", 30),
]

passed = 0
for text, category, threshold in alignment_tests:
    X = align_vectorizer.transform([text])
    prob = align_ensemble.predict_proba(X)[0][1] * 100
    
    if (category == "Professional" and prob >= threshold) or (category == "Poor" and prob < threshold):
        passed += 1
        status = "✅ PASS"
    else:
        status = "❌ FAIL"
    
    print(f"{category:12}: {prob:5.1f}% (Target: {'≥' if category == 'Professional' else '<'}{threshold}%) {status}")

align_accuracy = (passed / len(alignment_tests)) * 100
print(f"\nAlignment Test Accuracy: {align_accuracy:.1f}%")

# Test wellbeing
print("\n[3/3] Testing Wellbeing Model...")
print("-" * 40)

wellbeing_tests = [
    ("I want to end my life", "Crisis", 0.2),
    ("I'm so depressed", "Depression", 0.3),
    ("I'm feeling good today", "Positive", 0.7),
    ("Life is amazing!", "Euphoric", 0.8),
]

passed = 0
for text, category, threshold in wellbeing_tests:
    X = wellbeing_vectorizer.transform([text])
    prob = wellbeing_ensemble.predict_proba(X)[0][1]
    
    if (category in ["Positive", "Euphoric"] and prob >= threshold) or \
       (category in ["Crisis", "Depression"] and prob < threshold):
        passed += 1
        status = "✅ PASS"
    else:
        status = "❌ FAIL"
    
    print(f"{category:12}: {prob:5.2f} (Target: {'≥' if category in ['Positive', 'Euphoric'] else '<'}{threshold:.1f}) {status}")

wellbeing_accuracy = (passed / len(wellbeing_tests)) * 100
print(f"\nWellbeing Test Accuracy: {wellbeing_accuracy:.1f}%")

# Summary
print("\n" + "=" * 80)
print("V4 MODEL TEST SUMMARY")
print("=" * 80)
print(f"Alignment Accuracy: {align_accuracy:.1f}%")
print(f"Wellbeing Accuracy: {wellbeing_accuracy:.1f}%")
print(f"Overall: {(align_accuracy + wellbeing_accuracy) / 2:.1f}%")

if align_accuracy >= 90 and wellbeing_accuracy >= 90:
    print("\n🔥🔥🔥 V4 MODELS READY FOR WORLD DOMINATION! 🔥🔥🔥")
elif align_accuracy >= 80 and wellbeing_accuracy >= 80:
    print("\n✨ V4 models performing well!")
else:
    print("\n⚡ V4 models need more tuning")

print("=" * 80)