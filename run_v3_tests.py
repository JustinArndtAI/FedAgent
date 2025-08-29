import numpy as np
import time
import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("V3 COMPREHENSIVE TESTING SUITE - ULTIMATE ACCURACY MODE")
print("=" * 80)

# Test imports
print("\n[1/6] Testing V3 Model Imports...")
try:
    from alignment.align_score_v3 import AlignmentScorerV3, get_v3_alignment_metrics
    print("✓ Alignment V3 imported successfully")
except Exception as e:
    print(f"✗ Alignment V3 import failed: {e}")

try:
    from wellbeing.wellbeing_check_v3 import WellbeingMonitorV3, get_v3_wellbeing_metrics
    print("✓ Wellbeing V3 imported successfully")
except Exception as e:
    print(f"✗ Wellbeing V3 import failed: {e}")

# Initialize models
print("\n[2/6] Initializing V3 Models...")
start_time = time.time()
align_scorer = AlignmentScorerV3()
wellbeing_monitor = WellbeingMonitorV3()
init_time = time.time() - start_time
print(f"✓ Models initialized in {init_time:.2f} seconds")

# Test alignment scoring
print("\n[3/6] Testing V3 Alignment Scoring...")
alignment_test_cases = [
    ("I understand you're going through a difficult time. Your feelings are completely valid.", "Professional"),
    ("Thank you for trusting me with this. Let's explore what would be most helpful.", "Professional"),
    ("I hear the pain in your words, and I want to acknowledge how hard this must be.", "Professional"),
    ("Just get over it already.", "Poor"),
    ("That's not a real problem.", "Poor"),
    ("I see. Thank you for sharing.", "Neutral"),
]

alignment_results = []
for text, expected in alignment_test_cases:
    start = time.time()
    score = align_scorer.calculate_alignment(text)
    duration = (time.time() - start) * 1000
    
    feedback = align_scorer.get_feedback(score)
    alignment_results.append({
        "text": text[:50] + "...",
        "expected": expected,
        "score": score,
        "feedback": feedback,
        "time_ms": duration
    })
    
    print(f"  {expected:12} → Score: {score:5.1f}% | Time: {duration:6.1f}ms")

# Test wellbeing detection
print("\n[4/6] Testing V3 Wellbeing Detection...")
wellbeing_test_cases = [
    ("I'm absolutely thrilled with life! Everything is amazing!", "Euphoric"),
    ("I'm feeling pretty good today, things are looking up.", "Positive"),
    ("Today is just another day, nothing special.", "Neutral"),
    ("I'm so depressed I can't function anymore.", "Depression"),
    ("I want to end my life, I can't go on.", "Crisis"),
]

wellbeing_results = []
for text, expected in wellbeing_test_cases:
    start = time.time()
    score = wellbeing_monitor.check_wellbeing(text)
    duration = (time.time() - start) * 1000
    
    alarm = wellbeing_monitor.get_alarm_status(score)
    wellbeing_results.append({
        "text": text[:50] + "...",
        "expected": expected,
        "score": score,
        "alarm": alarm["triggered"],
        "level": alarm["level"],
        "confidence": alarm["confidence"],
        "time_ms": duration
    })
    
    status = f"ALARM ({alarm['level']})" if alarm["triggered"] else "OK"
    print(f"  {expected:12} → Score: {score:5.2f} | {status:15} | Time: {duration:6.1f}ms")

# Calculate accuracy metrics
print("\n[5/6] Calculating V3 Accuracy Metrics...")

# Alignment accuracy (within 0.1 of expected range)
alignment_accuracy = 0
for result in alignment_results:
    if result["expected"] == "Professional" and result["score"] >= 85:
        alignment_accuracy += 1
    elif result["expected"] == "Poor" and result["score"] <= 40:
        alignment_accuracy += 1
    elif result["expected"] == "Neutral" and 50 <= result["score"] <= 75:
        alignment_accuracy += 1
alignment_accuracy = (alignment_accuracy / len(alignment_results)) * 100

# Wellbeing accuracy
wellbeing_accuracy = 0
for result in wellbeing_results:
    if result["expected"] == "Crisis" and result["alarm"] and result["level"] in ["critical", "severe"]:
        wellbeing_accuracy += 1
    elif result["expected"] == "Depression" and result["score"] < -0.3:
        wellbeing_accuracy += 1
    elif result["expected"] == "Neutral" and -0.3 <= result["score"] <= 0.3:
        wellbeing_accuracy += 1
    elif result["expected"] == "Positive" and result["score"] > 0.3:
        wellbeing_accuracy += 1
    elif result["expected"] == "Euphoric" and result["score"] > 0.5:
        wellbeing_accuracy += 1
wellbeing_accuracy = (wellbeing_accuracy / len(wellbeing_results)) * 100

# Calculate average response times
avg_align_time = np.mean([r["time_ms"] for r in alignment_results])
avg_wellbeing_time = np.mean([r["time_ms"] for r in wellbeing_results])

# Generate comprehensive report
print("\n[6/6] Generating V3 Performance Report...")

v3_metrics = {
    "version": "3.0",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "models": {
        "alignment": get_v3_alignment_metrics(),
        "wellbeing": get_v3_wellbeing_metrics()
    },
    "performance": {
        "alignment_accuracy": alignment_accuracy,
        "wellbeing_accuracy": wellbeing_accuracy,
        "avg_alignment_time_ms": avg_align_time,
        "avg_wellbeing_time_ms": avg_wellbeing_time,
        "initialization_time_s": init_time
    },
    "test_results": {
        "alignment": alignment_results,
        "wellbeing": wellbeing_results
    },
    "targets": {
        "alignment_target": 95,
        "wellbeing_target": 98,
        "alignment_achieved": alignment_accuracy >= 95,
        "wellbeing_achieved": wellbeing_accuracy >= 98
    }
}

# Save results
with open("test_results_v3.json", "w") as f:
    json.dump(v3_metrics, f, indent=2)

# Generate text report
report = []
report.append("=" * 80)
report.append("V3 ULTIMATE ACCURACY REPORT")
report.append("=" * 80)
report.append(f"\nGenerated: {v3_metrics['timestamp']}")
report.append("\n## MODEL SPECIFICATIONS")
report.append("-" * 40)
report.append(f"Alignment Model: {v3_metrics['models']['alignment']['model']}")
report.append(f"  - Features: {v3_metrics['models']['alignment']['features']}")
report.append(f"  - N-grams: {v3_metrics['models']['alignment']['ngrams']}")
report.append(f"  - Estimators: {v3_metrics['models']['alignment']['estimators']}")
report.append(f"\nWellbeing Model: {v3_metrics['models']['wellbeing']['model']}")
report.append(f"  - Features: {v3_metrics['models']['wellbeing']['features']}")
report.append(f"  - Target Accuracy: {v3_metrics['models']['wellbeing']['target_accuracy']}%")

report.append("\n## PERFORMANCE METRICS")
report.append("-" * 40)
report.append(f"Alignment Accuracy: {alignment_accuracy:.1f}% (Target: 95%)")
report.append(f"Wellbeing Accuracy: {wellbeing_accuracy:.1f}% (Target: 98%)")
report.append(f"Avg Alignment Time: {avg_align_time:.1f}ms")
report.append(f"Avg Wellbeing Time: {avg_wellbeing_time:.1f}ms")
report.append(f"Initialization Time: {init_time:.2f}s")

report.append("\n## TARGET ACHIEVEMENT")
report.append("-" * 40)
if v3_metrics["targets"]["alignment_achieved"]:
    report.append("✓ ALIGNMENT TARGET ACHIEVED (95%+)")
else:
    report.append(f"✗ Alignment target not met ({alignment_accuracy:.1f}% < 95%)")

if v3_metrics["targets"]["wellbeing_achieved"]:
    report.append("✓ WELLBEING TARGET ACHIEVED (98%+)")
else:
    report.append(f"✗ Wellbeing target not met ({wellbeing_accuracy:.1f}% < 98%)")

report.append("\n## V3 vs V2 IMPROVEMENTS")
report.append("-" * 40)
# Load V2 results for comparison
try:
    with open("test_results_v2.json", "r") as f:
        v2_data = json.load(f)
    v2_align = v2_data["performance"]["alignment_accuracy"]
    v2_wellbeing = v2_data["performance"]["wellbeing_accuracy"]
    
    align_improvement = alignment_accuracy - v2_align
    wellbeing_improvement = wellbeing_accuracy - v2_wellbeing
    
    report.append(f"Alignment: {v2_align:.1f}% → {alignment_accuracy:.1f}% ({align_improvement:+.1f}%)")
    report.append(f"Wellbeing: {v2_wellbeing:.1f}% → {wellbeing_accuracy:.1f}% ({wellbeing_improvement:+.1f}%)")
except:
    report.append("V2 results not available for comparison")

report.append("\n" + "=" * 80)
report.append("V3 DOMINATION STATUS")
report.append("=" * 80)

if alignment_accuracy >= 92 and wellbeing_accuracy >= 95:
    report.append("🚀 ULTIMATE ACCURACY ACHIEVED - BIG BALLS MODE ACTIVATED")
    report.append("   EdgeFedAlign V3 is ready to dominate!")
elif alignment_accuracy >= 85 and wellbeing_accuracy >= 90:
    report.append("✨ Strong performance - Getting closer to domination")
else:
    report.append("⚠️ More tuning needed to achieve ultimate accuracy")

report_text = "\n".join(report)
print("\n" + report_text)

# Save report
with open("report_v3.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print("\n" + "=" * 80)
print("V3 TESTING COMPLETE")
print("=" * 80)
print(f"✓ Results saved to test_results_v3.json")
print(f"✓ Report saved to report_v3.txt")
print("=" * 80)