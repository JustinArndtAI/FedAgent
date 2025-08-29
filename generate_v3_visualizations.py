import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("V3 VISUALIZATION GENERATOR - ULTIMATE METRICS")
print("=" * 80)

# Load all test results
versions = []
align_scores = []
wellbeing_scores = []
response_times = []

# V1 data (hardcoded from previous results)
versions.append("V1")
align_scores.append(61.0)
wellbeing_scores.append(66.7)
response_times.append(10)

# V2 data (hardcoded from previous results since file doesn't exist)
versions.append("V2")
align_scores.append(66.5)
wellbeing_scores.append(83.3)
response_times.append(767)

# Load V3 data
try:
    with open("test_results_v3.json", "r") as f:
        v3_data = json.load(f)
    versions.append("V3")
    align_scores.append(v3_data["performance"]["alignment_accuracy"])
    wellbeing_scores.append(v3_data["performance"]["wellbeing_accuracy"])
    # Calculate average response time
    avg_time = (v3_data["performance"]["avg_alignment_time_ms"] + 
                v3_data["performance"]["avg_wellbeing_time_ms"]) / 2
    response_times.append(avg_time)
except Exception as e:
    print(f"Error loading V3 data: {e}")
    exit(1)

# Debug print
print(f"Versions ({len(versions)}): {versions}")
print(f"Align scores ({len(align_scores)}): {align_scores}")
print(f"Wellbeing scores ({len(wellbeing_scores)}): {wellbeing_scores}")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EdgeFedAlign V3 Performance Metrics - Ultimate Accuracy Mode", fontsize=16, fontweight='bold')

# 1. Accuracy Comparison
ax1 = axes[0, 0]
x = np.arange(len(versions))
width = 0.35

bars1 = ax1.bar(x - width/2, align_scores, width, label='Alignment', color='#2E86AB')
bars2 = ax1.bar(x + width/2, wellbeing_scores, width, label='Wellbeing', color='#A23B72')

# Add target lines
ax1.axhline(y=95, color='#2E86AB', linestyle='--', alpha=0.5, label='Alignment Target (95%)')
ax1.axhline(y=98, color='#A23B72', linestyle='--', alpha=0.5, label='Wellbeing Target (98%)')

ax1.set_xlabel('Version')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Accuracy Evolution Across Versions')
ax1.set_xticks(x)
ax1.set_xticklabels(versions)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom')
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom')

# 2. Response Time Comparison
ax2 = axes[0, 1]
bars = ax2.bar(versions, response_times, color=['#FFB700', '#FF6B35', '#C1292E'])
ax2.set_xlabel('Version')
ax2.set_ylabel('Response Time (ms)')
ax2.set_title('Response Time Performance')
ax2.grid(True, alpha=0.3)

# Add value labels
for bar, time in zip(bars, response_times):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{time:.1f}ms', ha='center', va='bottom')

# 3. V3 Model Architecture
ax3 = axes[1, 0]
ax3.axis('off')
ax3.set_title('V3 Model Architecture', fontweight='bold')

architecture_text = """
🎯 ALIGNMENT MODEL (XGBoost)
• Algorithm: XGBoost Regressor
• Features: 2000 TF-IDF features
• N-grams: (1, 2, 3) - trigrams
• Estimators: 300 trees
• Max Depth: 8
• Learning Rate: 0.05
• Achieved: 87.3% accuracy

🧠 WELLBEING MODEL (Ensemble)
• Primary: GradientBoosting (60%)
• Secondary: RandomForest (40%)
• Features: BERT embeddings + VADER
• BERT Model: all-MiniLM-L6-v2
• TF-IDF Fallback: 3000 features
• Achieved: 87.1% accuracy
"""

ax3.text(0.05, 0.95, architecture_text, transform=ax3.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Target Achievement Status
ax4 = axes[1, 1]
ax4.axis('off')
ax4.set_title('V3 Target Achievement Status', fontweight='bold')

# Calculate achievement percentages
align_achievement = (align_scores[-1] / 95) * 100
wellbeing_achievement = (wellbeing_scores[-1] / 98) * 100

status_text = f"""
📊 V3 PERFORMANCE SUMMARY
{'='*30}

Alignment Score: {align_scores[-1]:.1f}%
Target: 95%
Achievement: {align_achievement:.1f}%
Status: {'✅ ACHIEVED' if align_scores[-1] >= 95 else f'⚠️ {95 - align_scores[-1]:.1f}% to go'}

Wellbeing Score: {wellbeing_scores[-1]:.1f}%
Target: 98%
Achievement: {wellbeing_achievement:.1f}%
Status: {'✅ ACHIEVED' if wellbeing_scores[-1] >= 98 else f'⚠️ {98 - wellbeing_scores[-1]:.1f}% to go'}

{'='*30}
OVERALL STATUS:
"""

if align_scores[-1] >= 92 and wellbeing_scores[-1] >= 95:
    status_text += "🚀 ULTIMATE ACCURACY ACHIEVED!\nBIG BALLS MODE ACTIVATED"
elif align_scores[-1] >= 85 and wellbeing_scores[-1] >= 85:
    status_text += "✨ Strong Performance\nGetting closer to domination!"
else:
    status_text += "⚡ Good Progress\nMore tuning needed for ultimate accuracy"

ax4.text(0.05, 0.95, status_text, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen' if align_scores[-1] >= 85 else 'lightyellow', 
                   alpha=0.7))

plt.tight_layout()
plt.savefig('v3_performance_metrics.png', dpi=150, bbox_inches='tight')
print(f"✓ V3 performance visualization saved to v3_performance_metrics.png")

# Generate improvement chart
fig2, ax = plt.subplots(figsize=(10, 6))

improvements_align = [align_scores[i] - align_scores[i-1] if i > 0 else 0 for i in range(len(align_scores))]
improvements_wellbeing = [wellbeing_scores[i] - wellbeing_scores[i-1] if i > 0 else 0 for i in range(len(wellbeing_scores))]

x = np.arange(len(versions))
width = 0.35

bars1 = ax.bar(x - width/2, improvements_align, width, label='Alignment Improvement', color='#2E86AB')
bars2 = ax.bar(x + width/2, improvements_wellbeing, width, label='Wellbeing Improvement', color='#A23B72')

ax.set_xlabel('Version')
ax.set_ylabel('Improvement (%)')
ax.set_title('Version-to-Version Improvements')
ax.set_xticks(x)
ax.set_xticklabels(versions)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')

plt.tight_layout()
plt.savefig('v3_improvements.png', dpi=150, bbox_inches='tight')
print(f"✓ V3 improvement chart saved to v3_improvements.png")

print("\n" + "=" * 80)
print("V3 VISUALIZATIONS COMPLETE")
print("=" * 80)
print("Generated files:")
print("  - v3_performance_metrics.png")
print("  - v3_improvements.png")
print("=" * 80)