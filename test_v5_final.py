#!/usr/bin/env python
"""
V5 FINAL COMPREHENSIVE TEST SUITE
Complete testing with detailed logging and metrics
"""
import joblib
import numpy as np
import json
import time
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("🔥🔥🔥 V5 FINAL TEST SUITE - WORLD DOMINATION VERIFICATION 🔥🔥🔥")
print("=" * 80)

# Initialize test log
test_log = []
test_results = {
    "version": "5.0 FINAL",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "tests": [],
    "metrics": {}
}

# Load models
print("\n[1/5] Loading V5 Models...")
try:
    align_ensemble = joblib.load('v5_align_ensemble.pkl')
    align_vectorizer = joblib.load('v5_align_vectorizer.pkl')
    wellbeing_ensemble = joblib.load('v5_wellbeing_ensemble.pkl')
    wellbeing_vectorizer = joblib.load('v5_wellbeing_vectorizer.pkl')
    print("✅ All V5 models loaded successfully!")
    test_log.append(f"[{time.strftime('%H:%M:%S')}] Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    test_log.append(f"[{time.strftime('%H:%M:%S')}] ERROR: {e}")
    exit(1)

# COMPREHENSIVE ALIGNMENT TESTS
print("\n[2/5] Running Alignment Tests...")
print("-" * 60)

alignment_test_suite = {
    "Professional Responses": [
        ("I understand you're going through a difficult time. Your feelings are completely valid.", 95),
        ("Thank you for trusting me with this. Let's explore what would be most helpful.", 95),
        ("I hear the pain in your words, and I want to acknowledge how hard this must be.", 95),
        ("Your wellbeing is my priority. Together, we can work through this.", 95),
        ("It takes courage to share these feelings. You're not alone.", 95),
        ("I appreciate your openness. This is a safe space.", 90),
        ("What you're experiencing is more common than you might think.", 85),
    ],
    "Poor Responses": [
        ("Just get over it already.", 10),
        ("That's not a real problem.", 10),
        ("Stop being so dramatic.", 10),
        ("You're weak for feeling this way.", 5),
        ("Nobody cares about your issues.", 5),
        ("Man up and deal with it.", 10),
        ("You're making a big deal out of nothing.", 15),
    ],
    "Neutral Responses": [
        ("I see what you're saying.", 50),
        ("Thank you for sharing.", 60),
        ("That's interesting.", 45),
    ]
}

alignment_results = []
total_passed = 0
total_tests = 0

for category, tests in alignment_test_suite.items():
    print(f"\n{category}:")
    for text, threshold in tests:
        X = align_vectorizer.transform([text])
        prob = align_ensemble.predict_proba(X)[0][1] * 100
        
        passed = prob >= threshold if "Professional" in category else prob <= threshold
        status = "✅ PASS" if passed else "❌ FAIL"
        
        if passed:
            total_passed += 1
        total_tests += 1
        
        result = {
            "category": category,
            "text": text[:60] + "...",
            "score": float(prob),
            "threshold": threshold,
            "passed": bool(passed)
        }
        alignment_results.append(result)
        
        print(f"  [{status}] Score: {prob:6.1f}% (Target: {'≥' if 'Professional' in category else '≤'}{threshold}%)")
        print(f"        {text[:60]}...")
        
        test_log.append(f"[{time.strftime('%H:%M:%S')}] Alignment - {category}: {prob:.1f}% - {status}")

alignment_accuracy = (total_passed / total_tests) * 100

# COMPREHENSIVE WELLBEING TESTS
print("\n[3/5] Running Wellbeing Tests...")
print("-" * 60)

wellbeing_test_suite = {
    "Crisis": [
        ("I want to end my life", 0.2),
        ("I want to kill myself", 0.2),
        ("I can't go on anymore", 0.25),
        ("Life has no meaning anymore", 0.3),
        ("I'm planning to hurt myself", 0.2),
    ],
    "Depression": [
        ("I'm so depressed", 0.35),
        ("I'm so depressed I can't function", 0.3),
        ("Everything feels hopeless", 0.3),
        ("I feel empty inside", 0.35),
        ("Nothing brings me joy anymore", 0.35),
    ],
    "Anxiety": [
        ("I'm constantly anxious", 0.4),
        ("I can't stop worrying", 0.45),
        ("I'm having panic attacks", 0.35),
    ],
    "Positive": [
        ("I'm feeling good today", 0.7),
        ("I'm feeling pretty good today", 0.75),
        ("Things are looking up", 0.7),
        ("I'm optimistic about the future", 0.75),
        ("I'm happy with my progress", 0.7),
    ],
    "Euphoric": [
        ("Life is amazing!", 0.85),
        ("Everything is absolutely amazing!", 0.9),
        ("I'm on top of the world!", 0.85),
        ("Life couldn't be better!", 0.9),
    ]
}

wellbeing_results = []
wb_passed = 0
wb_total = 0

for category, tests in wellbeing_test_suite.items():
    print(f"\n{category}:")
    for text, threshold in tests:
        X = wellbeing_vectorizer.transform([text])
        prob = wellbeing_ensemble.predict_proba(X)[0][1]
        
        if category in ["Crisis", "Depression", "Anxiety"]:
            passed = prob <= threshold
        else:
            passed = prob >= threshold
        
        status = "✅ PASS" if passed else "❌ FAIL"
        
        if passed:
            wb_passed += 1
        wb_total += 1
        
        result = {
            "category": category,
            "text": text,
            "score": float(prob),
            "threshold": threshold,
            "passed": bool(passed)
        }
        wellbeing_results.append(result)
        
        print(f"  [{status}] Score: {prob:6.2f} (Target: {'≤' if category in ['Crisis', 'Depression', 'Anxiety'] else '≥'}{threshold:.2f})")
        print(f"        {text}")
        
        test_log.append(f"[{time.strftime('%H:%M:%S')}] Wellbeing - {category}: {prob:.2f} - {status}")

wellbeing_accuracy = (wb_passed / wb_total) * 100

# PERFORMANCE METRICS
print("\n[4/5] Calculating Performance Metrics...")
print("-" * 60)

# Test response times
response_times = []
test_texts = ["I need help", "I'm feeling anxious", "Life is good", "Thank you for listening"]

for text in test_texts:
    # Alignment timing
    start = time.time()
    X = align_vectorizer.transform([text])
    _ = align_ensemble.predict_proba(X)
    align_time = (time.time() - start) * 1000
    
    # Wellbeing timing
    start = time.time()
    X = wellbeing_vectorizer.transform([text])
    _ = wellbeing_ensemble.predict_proba(X)
    wb_time = (time.time() - start) * 1000
    
    response_times.append(align_time + wb_time)

avg_response_time = np.mean(response_times)

metrics = {
    "alignment_accuracy": alignment_accuracy,
    "wellbeing_accuracy": wellbeing_accuracy,
    "overall_accuracy": (alignment_accuracy + wellbeing_accuracy) / 2,
    "avg_response_time_ms": avg_response_time,
    "total_tests": total_tests + wb_total,
    "tests_passed": total_passed + wb_passed
}

print(f"Alignment Accuracy: {alignment_accuracy:.1f}% ({total_passed}/{total_tests} passed)")
print(f"Wellbeing Accuracy: {wellbeing_accuracy:.1f}% ({wb_passed}/{wb_total} passed)")
print(f"Overall Accuracy: {metrics['overall_accuracy']:.1f}%")
print(f"Average Response Time: {avg_response_time:.2f}ms")

# GENERATE COMPREHENSIVE REPORT
print("\n[5/5] Generating Reports and Logs...")
print("-" * 60)

# Save test results
test_results["metrics"] = metrics
test_results["alignment_tests"] = alignment_results
test_results["wellbeing_tests"] = wellbeing_results

with open("v5_test_results.json", "w") as f:
    json.dump(test_results, f, indent=2)
print("✓ Saved v5_test_results.json")

# Save test log
with open("v5_test_log.txt", "w", encoding="utf-8") as f:
    f.write("V5 FINAL TEST LOG\n")
    f.write("=" * 60 + "\n")
    f.write(f"Timestamp: {test_results['timestamp']}\n")
    f.write(f"Version: 5.0 FINAL\n")
    f.write("=" * 60 + "\n\n")
    for log_entry in test_log:
        f.write(log_entry + "\n")
    f.write("\n" + "=" * 60 + "\n")
    f.write(f"Final Results:\n")
    f.write(f"  Alignment: {alignment_accuracy:.1f}%\n")
    f.write(f"  Wellbeing: {wellbeing_accuracy:.1f}%\n")
    f.write(f"  Overall: {metrics['overall_accuracy']:.1f}%\n")
print("✓ Saved v5_test_log.txt")

# Generate final report
report_text = []
report_text.append("=" * 80)
report_text.append("V5 FINAL REPORT - WORLD DOMINATION STATUS")
report_text.append("=" * 80)
report_text.append(f"\nGenerated: {test_results['timestamp']}")
report_text.append(f"Version: 5.0 FINAL ULTIMATE")
report_text.append("\n## PERFORMANCE METRICS")
report_text.append("-" * 40)
report_text.append(f"Alignment Accuracy: {alignment_accuracy:.1f}%")
report_text.append(f"Wellbeing Accuracy: {wellbeing_accuracy:.1f}%")
report_text.append(f"Overall Accuracy: {metrics['overall_accuracy']:.1f}%")
report_text.append(f"Response Time: {avg_response_time:.2f}ms")
report_text.append(f"Tests Passed: {metrics['tests_passed']}/{metrics['total_tests']}")

if alignment_accuracy >= 98 and wellbeing_accuracy >= 99:
    report_text.append("\n🔥🔥🔥 WORLD DOMINATION ACHIEVED 🔥🔥🔥")
    report_text.append("ALL TARGETS OBLITERATED!")
elif alignment_accuracy >= 95 and wellbeing_accuracy >= 95:
    report_text.append("\n✨ EXCELLENT PERFORMANCE - NEAR DOMINATION")
else:
    report_text.append("\n⚡ GOOD PERFORMANCE")

report_text.append("\n## TEST BREAKDOWN")
report_text.append("-" * 40)
report_text.append(f"Alignment Tests: {total_tests} total, {total_passed} passed")
report_text.append(f"Wellbeing Tests: {wb_total} total, {wb_passed} passed")

report_text.append("\n" + "=" * 80)

report = "\n".join(report_text)
with open("v5_final_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("✓ Saved v5_final_report.txt")

# Display final summary
print("\n" + "=" * 80)
print("🔥🔥🔥 V5 FINAL TEST COMPLETE 🔥🔥🔥")
print("=" * 80)
print(f"Overall Accuracy: {metrics['overall_accuracy']:.1f}%")

if metrics['overall_accuracy'] >= 98:
    print("\n🌟🌟🌟 WORLD DOMINATION ACHIEVED - TARGETS OBLITERATED! 🌟🌟🌟")
    print("V5 IS READY FOR DEPLOYMENT!")

print("\nFiles Generated:")
print("  - v5_test_results.json")
print("  - v5_test_log.txt")
print("  - v5_final_report.txt")
print("=" * 80)