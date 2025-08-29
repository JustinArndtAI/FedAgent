#!/usr/bin/env python
"""
Finalize V6 and push to GitHub
Run this after training completes
"""
import os
import json
import subprocess
import time

def finalize_v6():
    """Check completion and push to GitHub"""
    print("=" * 60)
    print("V6 FINALIZATION SCRIPT")
    print("=" * 60)
    
    # Check if training is complete
    if not os.path.exists("v6_final_report.json"):
        print("\n[ERROR] Training not complete!")
        print("Run monitor_v6_training.py to check status")
        return False
    
    # Load and display results
    with open("v6_final_report.json", "r") as f:
        report = json.load(f)
    
    print(f"\nV6 FINAL RESULTS:")
    print(f"  Training time: {report['training_time_minutes']:.1f} minutes")
    print(f"  Samples: {report['samples_trained']}")
    print(f"  Alignment: {report['accuracies']['alignment']['validation']:.1f}%")
    print(f"  Wellbeing: {report['accuracies']['wellbeing']['validation']:.1f}%")
    print(f"  Overall: {report['accuracies']['overall']:.1f}%")
    
    # Check test results
    passed = sum(1 for r in report['test_results'] if r['result'] == 'PASS')
    total = len(report['test_results'])
    print(f"  Tests: {passed}/{total} passed")
    
    # Git operations
    print("\n[GIT] Staging files...")
    files_to_add = [
        "v6_final_*.pkl",
        "v6_final_*.json",
        "train_v6_final_50k.py",
        "monitor_v6_training.py",
        "finalize_v6.py"
    ]
    
    for pattern in files_to_add:
        subprocess.run(f"git add {pattern}", shell=True)
    
    # Create commit message
    commit_msg = f"""V6 FINAL BOSS - 50K Training Complete

Results:
- Alignment: {report['accuracies']['alignment']['validation']:.1f}% (Target: 98%)
- Wellbeing: {report['accuracies']['wellbeing']['validation']:.1f}% (Target: 99%)  
- Overall: {report['accuracies']['overall']:.1f}%
- Training time: {report['training_time_minutes']:.1f} minutes
- Test results: {passed}/{total} passed

Architecture:
- 50,000 training samples
- V4 base models + DistilBERT embeddings
- XGBoost + RandomForest + GradientBoosting ensemble
- 102-dimensional meta-features

Files:
- v6_final_align_ensemble.pkl
- v6_final_wellbeing_ensemble.pkl
- v6_final_report.json

Co-Authored-By: Claude <noreply@anthropic.com>"""
    
    # Commit
    print("\n[GIT] Creating commit...")
    with open("commit_msg.txt", "w") as f:
        f.write(commit_msg)
    
    result = subprocess.run("git commit -F commit_msg.txt", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("[OK] Commit created")
    else:
        print(f"[ERROR] Commit failed: {result.stderr}")
        return False
    
    # Push to GitHub
    print("\n[GIT] Pushing to GitHub...")
    result = subprocess.run("git push origin main", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("[OK] Pushed to GitHub successfully!")
    else:
        print(f"[ERROR] Push failed: {result.stderr}")
        return False
    
    # Clean up
    os.remove("commit_msg.txt")
    
    print("\n" + "=" * 60)
    print("V6 FINAL BOSS COMPLETE AND DEPLOYED!")
    print("=" * 60)
    
    if report['accuracies']['alignment']['validation'] >= 98 and report['accuracies']['wellbeing']['validation'] >= 99:
        print("\n*** TARGETS ACHIEVED! PRODUCTION READY! ***")
    elif report['accuracies']['overall'] >= 95:
        print("\n*** EXCELLENT PERFORMANCE! ***")
    
    print("\nGitHub: https://github.com/JustinArndtAI/FedAgent")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    finalize_v6()