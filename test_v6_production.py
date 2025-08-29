#!/usr/bin/env python
"""
V6 PRODUCTION TEST SUITE
Comprehensive testing of V6 FINAL BOSS models
"""
import requests
import json
import time

def test_v6_production():
    """Test V6 production API"""
    print("=" * 60)
    print("V6 PRODUCTION TEST SUITE")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        # Professional alignment
        ("I understand you're going through a difficult time.", "high", "positive"),
        ("Thank you for sharing this with me.", "high", "positive"),
        ("Your feelings are completely valid.", "high", "positive"),
        
        # Poor alignment
        ("Just get over it already.", "low", "negative"),
        ("Stop being so dramatic.", "low", "negative"),
        
        # Positive wellbeing
        ("I'm feeling great today!", "high", "positive"),
        ("Things are really looking up.", "high", "positive"),
        
        # Negative wellbeing
        ("I'm so depressed.", "low", "negative"),
        ("Everything feels hopeless.", "low", "negative"),
    ]
    
    print("\nStarting API tests...")
    base_url = "http://localhost:5000"
    
    # Health check
    try:
        r = requests.get(f"{base_url}/")
        print(f"Health check: {r.json()['status']}")
    except:
        print("[ERROR] API not running. Start with: python app.py")
        return
    
    # Test predictions
    passed = 0
    failed = 0
    
    for text, expected_align, expected_wb in test_cases:
        r = requests.post(f"{base_url}/predict", json={"text": text})
        result = r.json()
        
        align_pass = result['alignment_label'] == expected_align
        wb_pass = result['wellbeing_label'] == expected_wb
        
        if align_pass and wb_pass:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
        
        print(f"\n[{status}] {text[:50]}...")
        print(f"  Alignment: {result['alignment_score']:.2f} ({result['alignment_label']})")
        print(f"  Wellbeing: {result['wellbeing_score']:.2f} ({result['wellbeing_label']})")
    
    # Batch test
    print("\nTesting batch endpoint...")
    texts = ["I'm happy", "I'm sad", "Thank you"]
    r = requests.post(f"{base_url}/batch", json={"texts": texts})
    batch_result = r.json()
    print(f"Batch processed: {batch_result['total']} texts in {batch_result['processing_time_ms']}ms")
    
    # Final report
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{passed+failed} tests passed")
    if passed == len(test_cases):
        print("*** ALL TESTS PASSED! PRODUCTION READY! ***")
    print("=" * 60)

if __name__ == "__main__":
    test_v6_production()
