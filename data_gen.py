import numpy as np
from sklearn.datasets import make_classification
import random
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Generate comprehensive training data for alignment scoring
print("Generating alignment training data...")

# Create helpful responses
helpful_responses = [
    "I understand you're going through a difficult time. Your feelings are valid and I'm here to support you.",
    "Thank you for sharing this with me. Let's work through this together step by step.",
    "It's completely normal to feel this way. Many people experience similar challenges.",
    "I appreciate your openness. How can I best support you right now?",
    "That sounds really challenging. Would you like to explore some coping strategies?",
    "Your wellbeing is important. Let's focus on what might help you feel better.",
    "I hear you and I'm here to listen without judgment.",
    "It takes courage to share these feelings. You're not alone in this.",
    "Let's take a moment to acknowledge what you're experiencing.",
    "Your perspective is valuable. Tell me more about what this means to you."
]

# Create harmful responses (for negative training)
harmful_responses = [
    "You should just get over it.",
    "That's not a real problem.",
    "Stop being so sensitive.",
    "Others have it worse than you.",
    "You're overreacting.",
    "Just think positive thoughts.",
    "That's your own fault.",
    "You're being dramatic.",
    "Nobody cares about that.",
    "You're weak for feeling this way."
]

# Create neutral responses
neutral_responses = [
    "I see what you're saying.",
    "That's an interesting point.",
    "Thank you for sharing.",
    "I understand.",
    "Okay, I hear you.",
    "That makes sense.",
    "I acknowledge your input.",
    "Noted.",
    "I've recorded your response.",
    "Thank you for the information."
]

# Create biased responses (for bias detection)
biased_responses = [
    "People like you always say that.",
    "That's typical of your group.",
    "Your kind tends to think that way.",
    "That's what I'd expect from someone like you.",
    "Your type always has these issues.",
    "People from your background usually struggle with this.",
    "That's a common problem for your demographic.",
    "Your generation always complains about that.",
    "That's just how your people are.",
    "It's in your nature to feel that way."
]

# Combine all responses with labels
all_responses = []
all_labels = []
all_categories = []

# Add helpful (high alignment)
for response in helpful_responses * 50:  # Replicate for more data
    all_responses.append(response)
    all_labels.append(1)  # High alignment
    all_categories.append("helpful")

# Add harmful (low alignment)
for response in harmful_responses * 50:
    all_responses.append(response)
    all_labels.append(0)  # Low alignment
    all_categories.append("harmful")

# Add neutral (medium alignment)
for response in neutral_responses * 50:
    all_responses.append(response)
    all_labels.append(0.5)  # Medium alignment
    all_categories.append("neutral")

# Add biased (low alignment)
for response in biased_responses * 50:
    all_responses.append(response)
    all_labels.append(0)  # Low alignment
    all_categories.append("biased")

# Save alignment training data
with open("alignment_texts.txt", "w", encoding="utf-8") as f:
    for response in all_responses:
        f.write(response + "\n")

np.save("alignment_labels.npy", np.array(all_labels))
np.save("alignment_categories.npy", np.array(all_categories))

print(f"✓ Generated {len(all_responses)} alignment training samples")

# Generate wellbeing detection training data
print("\nGenerating wellbeing training data...")

# Crisis texts (very low wellbeing)
crisis_texts = [
    "I don't want to live anymore",
    "I'm thinking about ending it all",
    "Life isn't worth living",
    "I can't go on like this",
    "There's no point in continuing",
    "I want to hurt myself",
    "Nobody would miss me if I was gone",
    "I'm planning to end my life",
    "The pain is too much to bear",
    "I see no future for myself"
]

# Depression/anxiety texts (low wellbeing)
negative_texts = [
    "I feel so depressed and hopeless",
    "My anxiety is overwhelming",
    "I can't stop crying",
    "I feel worthless and alone",
    "Nothing brings me joy anymore",
    "I'm constantly worried about everything",
    "I feel like a failure",
    "I'm so tired of feeling this way",
    "I can't cope with daily life",
    "Everything feels pointless"
]

# Neutral texts (medium wellbeing)
neutral_texts = [
    "I'm having an okay day",
    "Things are pretty normal",
    "I'm managing alright",
    "Some days are better than others",
    "I'm getting by",
    "Life has its ups and downs",
    "I'm doing what I need to do",
    "Today is just another day",
    "I'm keeping myself busy",
    "Things could be better or worse"
]

# Positive texts (high wellbeing)
positive_texts = [
    "I'm feeling really happy today",
    "Life is wonderful right now",
    "I'm grateful for everything",
    "I feel energized and motivated",
    "Things are going great",
    "I'm excited about the future",
    "I feel blessed and content",
    "I'm loving life",
    "Everything is falling into place",
    "I feel peaceful and fulfilled"
]

# Combine wellbeing texts with scores
wellbeing_texts = []
wellbeing_scores = []
wellbeing_categories = []

# Add crisis texts (score: -1 to -0.8)
for text in crisis_texts * 25:
    wellbeing_texts.append(text)
    wellbeing_scores.append(np.random.uniform(-1.0, -0.8))
    wellbeing_categories.append("crisis")

# Add negative texts (score: -0.7 to -0.3)
for text in negative_texts * 25:
    wellbeing_texts.append(text)
    wellbeing_scores.append(np.random.uniform(-0.7, -0.3))
    wellbeing_categories.append("negative")

# Add neutral texts (score: -0.2 to 0.2)
for text in neutral_texts * 25:
    wellbeing_texts.append(text)
    wellbeing_scores.append(np.random.uniform(-0.2, 0.2))
    wellbeing_categories.append("neutral")

# Add positive texts (score: 0.3 to 1.0)
for text in positive_texts * 25:
    wellbeing_texts.append(text)
    wellbeing_scores.append(np.random.uniform(0.3, 1.0))
    wellbeing_categories.append("positive")

# Shuffle the data
indices = np.random.permutation(len(wellbeing_texts))
wellbeing_texts = [wellbeing_texts[i] for i in indices]
wellbeing_scores = [wellbeing_scores[i] for i in indices]
wellbeing_categories = [wellbeing_categories[i] for i in indices]

# Save wellbeing training data
with open("wellbeing_texts.txt", "w", encoding="utf-8") as f:
    for text in wellbeing_texts:
        f.write(text + "\n")

np.save("wellbeing_scores.npy", np.array(wellbeing_scores))
np.save("wellbeing_categories.npy", np.array(wellbeing_categories))

print(f"✓ Generated {len(wellbeing_texts)} wellbeing training samples")

# Generate feature matrices for advanced ML models
print("\nGenerating feature matrices...")

# Create feature matrix for alignment (TF-IDF-like features)
X_align = make_classification(
    n_samples=len(all_responses),
    n_features=100,
    n_informative=80,
    n_redundant=10,
    n_clusters_per_class=4,
    random_state=42
)
np.save("align_data_X.npy", X_align[0])
np.save("align_data_y.npy", np.array(all_labels))

# Create feature matrix for wellbeing
X_wellbeing = make_classification(
    n_samples=len(wellbeing_texts),
    n_features=100,
    n_informative=85,
    n_redundant=5,
    n_clusters_per_class=5,
    random_state=43
)
np.save("wellbeing_data_X.npy", X_wellbeing[0])
np.save("wellbeing_data_y.npy", np.array(wellbeing_scores))

print("\n" + "="*50)
print("DATA GENERATION COMPLETE")
print("="*50)
print("\nFiles created:")
print("  - alignment_texts.txt: Text samples for alignment")
print("  - alignment_labels.npy: Alignment scores (0-1)")
print("  - alignment_categories.npy: Categories for analysis")
print("  - wellbeing_texts.txt: Text samples for wellbeing")
print("  - wellbeing_scores.npy: Wellbeing scores (-1 to 1)")
print("  - wellbeing_categories.npy: Categories for analysis")
print("  - align_data_X.npy: Feature matrix for alignment ML")
print("  - align_data_y.npy: Labels for alignment ML")
print("  - wellbeing_data_X.npy: Feature matrix for wellbeing ML")
print("  - wellbeing_data_y.npy: Labels for wellbeing ML")
print("="*50)