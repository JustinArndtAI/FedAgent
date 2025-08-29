#!/usr/bin/env python
"""
Monitor V6 training progress
"""
import os
import time
import json

def check_progress():
    """Check V6 training status"""
    print("=" * 60)
    print("V6 FINAL BOSS TRAINING MONITOR")
    print("=" * 60)
    
    # Check if training is complete
    if os.path.exists("v6_final_report.json"):
        print("\n[COMPLETE] Training finished!")
        with open("v6_final_report.json", "r") as f:
            report = json.load(f)
        
        print(f"\nTraining time: {report['training_time_minutes']:.1f} minutes")
        print(f"\nAccuracies:")
        print(f"  Alignment: {report['accuracies']['alignment']['validation']:.1f}%")
        print(f"  Wellbeing: {report['accuracies']['wellbeing']['validation']:.1f}%")
        print(f"  Overall: {report['accuracies']['overall']:.1f}%")
        
        passed = sum(1 for r in report['test_results'] if r['result'] == 'PASS')
        total = len(report['test_results'])
        print(f"\nTest results: {passed}/{total} passed")
        
        if report['accuracies']['alignment']['achieved'] and report['accuracies']['wellbeing']['achieved']:
            print("\n*** TARGETS ACHIEVED! ***")
        
        return True
    
    # Check for model files
    models = [
        "v6_final_align_ensemble.pkl",
        "v6_final_wellbeing_ensemble.pkl"
    ]
    
    existing = [m for m in models if os.path.exists(m)]
    if existing:
        print(f"\n[IN PROGRESS] Models saved: {len(existing)}/{len(models)}")
        for m in existing:
            size = os.path.getsize(m) / (1024 * 1024)
            print(f"  - {m}: {size:.1f} MB")
    else:
        print("\n[IN PROGRESS] Training in progress...")
        print("  Generating 50,000 samples")
        print("  Extracting DistilBERT embeddings")
        print("  Training ensemble models")
        print("\nThis will take 20-30 minutes on CPU")
        print("Check back in 10 minutes...")
    
    return False

if __name__ == "__main__":
    complete = check_progress()
    
    if complete:
        print("\nNext steps:")
        print("  1. Review v6_final_report.json")
        print("  2. Test with app.py")
        print("  3. Push to GitHub")
    else:
        print("\nRun this script again to check progress:")
        print("  python monitor_v6_training.py")
    
    print("=" * 60)