#!/usr/bin/env python
"""
V4 TRAINING STATUS CHECKER
Check if V4 models are trained and ready for world domination
"""
import os
import sys
import time
import json
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("🔥 V4 FINAL BOSS STATUS CHECK 🔥")
print("=" * 80)

# Check for required model files
required_files = {
    "v4_align_ensemble.pkl": "Alignment ensemble model",
    "v4_align_vectorizer.pkl": "Alignment TF-IDF vectorizer",
    "v4_align_xgb.json": "Alignment XGBoost model",
    "v4_wellbeing_ensemble.pkl": "Wellbeing ensemble model",
    "v4_wellbeing_vectorizer.pkl": "Wellbeing TF-IDF vectorizer",
    "v4_wellbeing_xgb.json": "Wellbeing XGBoost model"
}

# Check data files (should already exist)
data_files = {
    "v4_align_texts.npy": "50K alignment texts",
    "v4_align_labels.npy": "Alignment labels",
    "v4_wellbeing_texts.txt": "50K wellbeing texts",
    "v4_wellbeing_scores.npy": "Wellbeing scores"
}

print("\n[1/3] Checking Training Data...")
print("-" * 40)
data_ready = True
for file, desc in data_files.items():
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"✓ {file}: {desc} ({size_mb:.1f} MB)")
    else:
        print(f"✗ {file}: Missing!")
        data_ready = False

if data_ready:
    print("\n✓ Training data ready (100K samples)")
else:
    print("\n✗ Training data incomplete - run: python data_gen_v4.py")

print("\n[2/3] Checking Trained Models...")
print("-" * 40)
models_ready = True
trained_count = 0
for file, desc in required_files.items():
    if os.path.exists(file):
        size_kb = os.path.getsize(file) / 1024
        print(f"✓ {file}: {desc} ({size_kb:.1f} KB)")
        trained_count += 1
    else:
        print(f"✗ {file}: Not found - {desc}")
        models_ready = False

progress = (trained_count / len(required_files)) * 100
print(f"\nTraining Progress: {progress:.0f}% ({trained_count}/{len(required_files)} models)")

print("\n[3/3] Testing Model Loading...")
print("-" * 40)

if models_ready:
    try:
        # Try to import and load V4 models
        from alignment.align_score_v4 import AlignmentScorerV4
        from wellbeing.wellbeing_check_v4 import WellbeingMonitorV4
        
        print("Testing alignment model...")
        align_scorer = AlignmentScorerV4()
        test_score = align_scorer.calculate_alignment("Thank you for sharing this with me.")
        print(f"✓ Alignment model working: {test_score:.1f}%")
        
        print("Testing wellbeing model...")
        wellbeing_monitor = WellbeingMonitorV4()
        test_score = wellbeing_monitor.check_wellbeing("I'm feeling good today.")
        print(f"✓ Wellbeing model working: {test_score:.2f}")
        
        print("\n" + "=" * 80)
        print("🔥🔥🔥 V4 TRAINING COMPLETE - WORLD DOMINATION READY! 🔥🔥🔥")
        print("=" * 80)
        print("\nYou can now run:")
        print("  python main_v4.py --demo      # Interactive FINAL BOSS mode")
        print("  python run_v4_tests.py        # Brutal accuracy tests")
        
    except Exception as e:
        print(f"⚠️ Models exist but failed to load: {e}")
        print("\nModels may still be training. Try again in a few minutes.")
        
elif trained_count > 0:
    print("\n⏳ TRAINING IN PROGRESS...")
    print(f"   {trained_count} of {len(required_files)} models complete")
    print("\nEstimated time remaining:")
    print("  - CPU: 10-20 minutes")
    print("  - GPU: 2-5 minutes")
    print("\nCheck again in a few minutes.")
    
else:
    print("\n🔥 TRAINING NOT STARTED")
    print("\nTo train V4 models, run these commands:")
    print("  1. python data_gen_v4.py          # Generate 100K samples (✓ Complete)")
    print("  2. python alignment/align_score_v4.py    # Train alignment model")
    print("  3. python wellbeing/wellbeing_check_v4.py # Train wellbeing model")
    print("\nOr use the automated trainer:")
    print("  python main_v4.py --train")
    print("\nNote: Training will take 15-30 minutes on CPU due to:")
    print("  - 50K DistilBERT embeddings extraction")
    print("  - Optuna hyperparameter optimization (30+ trials)")
    print("  - Triple ensemble training")

# Check for test results
print("\n" + "=" * 80)
print("Test Results Status:")
print("-" * 40)

if os.path.exists("test_results_v4.json"):
    with open("test_results_v4.json", "r") as f:
        results = json.load(f)
    
    print(f"✓ Last test run: {results.get('timestamp', 'Unknown')}")
    perf = results.get('performance', {})
    print(f"  Alignment: {perf.get('alignment_accuracy', 'N/A')}%")
    print(f"  Wellbeing: {perf.get('wellbeing_accuracy', 'N/A')}%")
    
    targets = results.get('targets', {})
    if targets.get('world_domination'):
        print("\n🔥🔥🔥 WORLD DOMINATION ACHIEVED! 🔥🔥🔥")
else:
    print("✗ No test results found - run: python run_v4_tests.py")

print("\n" + "=" * 80)