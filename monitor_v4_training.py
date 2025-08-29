#!/usr/bin/env python
"""
V4 TRAINING MONITOR - Real-time progress tracking
"""
import os
import time
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_model_files():
    """Check which model files exist"""
    models = {
        "Alignment": [
            ("v4_align_ensemble.pkl", "Ensemble"),
            ("v4_align_vectorizer.pkl", "Vectorizer"),
            ("v4_align_xgb.json", "XGBoost")
        ],
        "Wellbeing": [
            ("v4_wellbeing_ensemble.pkl", "Ensemble"),
            ("v4_wellbeing_vectorizer.pkl", "Vectorizer"),
            ("v4_wellbeing_xgb.json", "XGBoost")
        ]
    }
    
    status = {}
    for category, files in models.items():
        status[category] = []
        for filename, desc in files:
            exists = os.path.exists(filename)
            size = os.path.getsize(filename) / 1024 if exists else 0
            status[category].append((desc, exists, size))
    
    return status

def main():
    print("=" * 80)
    print("🔥 V4 TRAINING MONITOR - LIVE STATUS 🔥")
    print("=" * 80)
    print("\nMonitoring V4 model training... (Press Ctrl+C to stop)")
    print("\nTraining Phases:")
    print("1. Extract TF-IDF features (1-2 min)")
    print("2. Extract DistilBERT embeddings (10-15 min)")
    print("3. Extract professional/pattern features (1 min)")
    print("4. Run Optuna optimization (5-10 min)")
    print("5. Train ensemble models (2-5 min)")
    print("6. Save models (< 1 min)")
    
    start_time = time.time()
    last_status = {}
    
    while True:
        try:
            # Clear screen (optional)
            # os.system('cls' if os.name == 'nt' else 'clear')
            
            current_status = check_model_files()
            elapsed = time.time() - start_time
            elapsed_min = elapsed / 60
            
            print(f"\n[{time.strftime('%H:%M:%S')}] Elapsed: {elapsed_min:.1f} minutes")
            print("-" * 60)
            
            # Check alignment models
            print("\n📊 ALIGNMENT MODEL STATUS:")
            align_complete = 0
            for desc, exists, size_kb in current_status["Alignment"]:
                symbol = "✅" if exists else "⏳"
                if exists:
                    align_complete += 1
                    print(f"  {symbol} {desc}: Ready ({size_kb:.1f} KB)")
                else:
                    print(f"  {symbol} {desc}: Training...")
            
            align_progress = (align_complete / 3) * 100
            print(f"  Progress: {align_progress:.0f}% ({align_complete}/3 files)")
            
            # Check wellbeing models
            print("\n🧠 WELLBEING MODEL STATUS:")
            well_complete = 0
            for desc, exists, size_kb in current_status["Wellbeing"]:
                symbol = "✅" if exists else "⏳"
                if exists:
                    well_complete += 1
                    print(f"  {symbol} {desc}: Ready ({size_kb:.1f} KB)")
                else:
                    print(f"  {symbol} {desc}: Training...")
            
            well_progress = (well_complete / 3) * 100
            print(f"  Progress: {well_progress:.0f}% ({well_complete}/3 files)")
            
            # Overall progress
            total_complete = align_complete + well_complete
            total_progress = (total_complete / 6) * 100
            
            print("\n" + "=" * 60)
            print(f"🔥 OVERALL PROGRESS: {total_progress:.0f}% ({total_complete}/6 models)")
            
            # Progress bar
            bar_length = 40
            filled = int(bar_length * total_progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"[{bar}] {total_progress:.0f}%")
            
            if total_progress == 100:
                print("\n" + "🎉" * 30)
                print("🔥🔥🔥 V4 TRAINING COMPLETE - WORLD DOMINATION READY! 🔥🔥🔥")
                print("🎉" * 30)
                print("\nAll models trained successfully!")
                print("\nNext steps:")
                print("  1. python run_v4_tests.py    # Run brutal accuracy tests")
                print("  2. python main_v4.py --demo   # Interactive FINAL BOSS mode")
                break
            
            # Estimate remaining time
            if total_complete > 0:
                time_per_model = elapsed / total_complete
                remaining_models = 6 - total_complete
                est_remaining = (time_per_model * remaining_models) / 60
                print(f"\nEstimated time remaining: {est_remaining:.1f} minutes")
            else:
                print("\nEstimated total time: 15-30 minutes on CPU")
            
            # Check for changes
            if current_status != last_status:
                if total_complete > sum(1 for cat in last_status.values() for _, exists, _ in cat if exists):
                    print("\n✨ New model completed!")
            
            last_status = current_status
            
            # Wait before next check
            time.sleep(10)  # Check every 10 seconds
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            print(f"Current progress: {total_progress:.0f}%")
            print("Run this script again to continue monitoring.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()