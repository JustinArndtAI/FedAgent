import numpy as np
import random
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("V3 DATA GENERATION - ULTIMATE DATASET")
print("=" * 80)

# Generate 5000+ ultra-realistic therapy/alignment samples
print("\nGenerating V3 Alignment Dataset (5000+ samples)...")

# Professional therapy responses (HIGH alignment - 95%+)
professional_responses = [
    "I understand this is incredibly difficult for you. Your feelings are completely valid, and I'm here to support you through this challenging time.",
    "Thank you for trusting me with this. Let's explore what would be most helpful for you right now.",
    "It takes tremendous courage to share these feelings. I want you to know that you're not alone in this journey.",
    "I hear the pain in your words, and I want to acknowledge how hard this must be for you.",
    "Your wellbeing is my priority. Let's work together to find strategies that resonate with you.",
    "I appreciate your openness. This is a safe space for you to express whatever you're feeling.",
    "What you're experiencing is more common than you might think, and there are effective ways we can address it together.",
    "I can see how much strength it takes to reach out. That's an important first step.",
    "Your perspective matters, and I'm here to listen without judgment.",
    "Let's take this one step at a time. There's no pressure to have everything figured out right now.",
    "I recognize the courage it takes to be vulnerable. Thank you for sharing this with me.",
    "It sounds like you've been carrying this burden for a while. How can I best support you?",
    "Your feelings are understandable given what you've been through. Let's explore healthy coping strategies.",
    "I want to make sure I'm understanding correctly. Can you tell me more about what this means for you?",
    "You deserve compassion and support. I'm committed to helping you work through this.",
]

# Empathetic variations
empathetic_additions = [
    " I truly care about your wellbeing.",
    " Your mental health matters.",
    " You are valued and important.",
    " This is a judgment-free zone.",
    " I'm here to listen and support you.",
    " Your experiences are valid.",
    " Let's find what works best for you.",
    " You don't have to face this alone.",
    " I believe in your resilience.",
    " Together, we can work through this.",
]

# Poor responses (LOW alignment - <50%)
poor_responses = [
    "Just get over it already.",
    "That's not really a problem.",
    "You're being too sensitive.",
    "Other people have it worse.",
    "Stop complaining.",
    "It's all in your head.",
    "You're making a big deal out of nothing.",
    "Just think positive.",
    "You need to toughen up.",
    "That's your own fault.",
    "Nobody wants to hear about your problems.",
    "You're being dramatic.",
    "Man up and deal with it.",
    "You're weak for feeling this way.",
    "Stop seeking attention.",
]

# Neutral responses (MEDIUM alignment - 60-75%)
neutral_responses = [
    "I see.",
    "Okay, I understand.",
    "Thank you for sharing.",
    "That's interesting.",
    "I hear what you're saying.",
    "Noted.",
    "I acknowledge your input.",
    "That's one perspective.",
    "I'll keep that in mind.",
    "Thanks for letting me know.",
]

# Build massive alignment dataset
alignment_texts = []
alignment_labels = []

# Add 2000 professional responses with variations
for _ in range(2000):
    base = random.choice(professional_responses)
    if random.random() > 0.5:
        base += random.choice(empathetic_additions)
    alignment_texts.append(base)
    alignment_labels.append(0.95 + random.uniform(0, 0.05))  # 95-100% alignment

# Add 1500 poor responses with variations
for _ in range(1500):
    response = random.choice(poor_responses)
    if random.random() > 0.7:
        response += " " + random.choice(["Whatever.", "Deal with it.", "Not my problem."])
    alignment_texts.append(response)
    alignment_labels.append(random.uniform(0.1, 0.4))  # 10-40% alignment

# Add 1000 neutral responses
for _ in range(1000):
    response = random.choice(neutral_responses)
    alignment_texts.append(response)
    alignment_labels.append(random.uniform(0.6, 0.75))  # 60-75% alignment

# Add 500 mixed quality responses
for _ in range(500):
    if random.random() > 0.5:
        # Mix good with neutral
        response = random.choice(professional_responses[:5]) + " " + random.choice(neutral_responses)
        alignment_texts.append(response)
        alignment_labels.append(random.uniform(0.75, 0.85))
    else:
        # Mix poor with neutral
        response = random.choice(neutral_responses) + " " + random.choice(poor_responses[:5])
        alignment_texts.append(response)
        alignment_labels.append(random.uniform(0.4, 0.6))

print(f"✓ Generated {len(alignment_texts)} alignment samples")

# Save alignment data
np.save("v3_align_texts.npy", np.array(alignment_texts))
np.save("v3_align_labels.npy", np.array(alignment_labels))

# Generate 5000+ wellbeing samples
print("\nGenerating V3 Wellbeing Dataset (5000+ samples)...")

# Crisis texts (very low wellbeing: -1.0 to -0.8)
crisis_texts = [
    "I want to end my life",
    "I can't go on anymore",
    "Life has no meaning",
    "I'm planning to hurt myself",
    "Nobody would care if I died",
    "I feel completely hopeless",
    "There's no point in living",
    "I want to disappear forever",
    "The pain is unbearable",
    "I've given up on everything",
]

# Depression texts (low wellbeing: -0.7 to -0.4)
depression_texts = [
    "I'm so depressed I can't function",
    "Everything feels pointless",
    "I hate myself so much",
    "I'm worthless and useless",
    "I can't stop crying",
    "I feel empty inside",
    "Nothing brings me joy anymore",
    "I'm drowning in sadness",
    "I feel completely alone",
    "My life is falling apart",
]

# Anxiety texts (medium-low wellbeing: -0.5 to -0.2)
anxiety_texts = [
    "I'm constantly anxious",
    "I can't stop worrying",
    "I'm having panic attacks",
    "Everything scares me",
    "I feel overwhelmed",
    "I'm stressed beyond belief",
    "I can't handle the pressure",
    "My anxiety is out of control",
    "I'm terrified of everything",
    "I feel like I'm losing my mind",
]

# Neutral texts (neutral wellbeing: -0.2 to 0.2)
neutral_texts = [
    "Today is just another day",
    "I'm doing okay I guess",
    "Things are fine",
    "Life goes on",
    "I'm managing",
    "It's been an average day",
    "Nothing special happening",
    "I'm getting by",
    "Same old routine",
    "Just taking it day by day",
]

# Positive texts (high wellbeing: 0.3 to 0.7)
positive_texts = [
    "I'm feeling pretty good today",
    "Things are looking up",
    "I'm happy with my progress",
    "Life is treating me well",
    "I feel content and peaceful",
    "I'm grateful for what I have",
    "Today was a good day",
    "I'm optimistic about the future",
    "I feel balanced and stable",
    "I'm in a good place mentally",
]

# Euphoric texts (very high wellbeing: 0.8 to 1.0)
euphoric_texts = [
    "I'm absolutely thrilled with life!",
    "Everything is amazing!",
    "I've never been happier!",
    "Life is absolutely wonderful!",
    "I'm on top of the world!",
    "This is the best day ever!",
    "I'm bursting with joy!",
    "Everything is perfect!",
    "I feel invincible!",
    "Life couldn't be better!",
]

wellbeing_texts = []
wellbeing_scores = []

# Add samples with realistic distribution
# Crisis: 500 samples
for _ in range(500):
    text = random.choice(crisis_texts)
    if random.random() > 0.5:
        text += " " + random.choice(["Please help me.", "I don't know what to do.", "I'm scared."])
    wellbeing_texts.append(text)
    wellbeing_scores.append(random.uniform(-1.0, -0.8))

# Depression: 1000 samples
for _ in range(1000):
    text = random.choice(depression_texts)
    if random.random() > 0.5:
        text += " " + random.choice(["Every day is a struggle.", "I can't see a way out.", "Nothing helps."])
    wellbeing_texts.append(text)
    wellbeing_scores.append(random.uniform(-0.7, -0.4))

# Anxiety: 1000 samples
for _ in range(1000):
    text = random.choice(anxiety_texts)
    wellbeing_texts.append(text)
    wellbeing_scores.append(random.uniform(-0.5, -0.2))

# Neutral: 1500 samples
for _ in range(1500):
    text = random.choice(neutral_texts)
    wellbeing_texts.append(text)
    wellbeing_scores.append(random.uniform(-0.2, 0.2))

# Positive: 800 samples
for _ in range(800):
    text = random.choice(positive_texts)
    if random.random() > 0.5:
        text += " " + random.choice(["Things are improving.", "I'm making progress.", "Feeling hopeful."])
    wellbeing_texts.append(text)
    wellbeing_scores.append(random.uniform(0.3, 0.7))

# Euphoric: 200 samples
for _ in range(200):
    text = random.choice(euphoric_texts)
    wellbeing_texts.append(text)
    wellbeing_scores.append(random.uniform(0.8, 1.0))

print(f"✓ Generated {len(wellbeing_texts)} wellbeing samples")

# Save wellbeing data
with open("v3_wellbeing_texts.txt", "w", encoding="utf-8") as f:
    for text in wellbeing_texts:
        f.write(text + "\n")

np.save("v3_wellbeing_scores.npy", np.array(wellbeing_scores))

# Create feature matrices for ML
print("\nGenerating feature matrices...")

from sklearn.datasets import make_classification

# Alignment features (more complex)
X_align = make_classification(
    n_samples=len(alignment_texts),
    n_features=200,
    n_informative=180,
    n_redundant=10,
    n_clusters_per_class=10,
    random_state=42
)

# Wellbeing features (more complex)
X_wellbeing = make_classification(
    n_samples=len(wellbeing_texts),
    n_features=200,
    n_informative=180,
    n_redundant=10,
    n_clusters_per_class=8,
    random_state=43
)

np.save("v3_align_features.npy", X_align[0])
np.save("v3_wellbeing_features.npy", X_wellbeing[0])

print("\n" + "=" * 80)
print("V3 DATA GENERATION COMPLETE")
print("=" * 80)
print(f"\nDataset Statistics:")
print(f"  - Alignment samples: {len(alignment_texts)}")
print(f"  - Wellbeing samples: {len(wellbeing_texts)}")
print(f"  - Feature dimensions: 200")
print(f"  - Total samples: {len(alignment_texts) + len(wellbeing_texts)}")
print("\nFiles created:")
print("  - v3_align_texts.npy")
print("  - v3_align_labels.npy")
print("  - v3_wellbeing_texts.txt")
print("  - v3_wellbeing_scores.npy")
print("  - v3_align_features.npy")
print("  - v3_wellbeing_features.npy")
print("\n✓ Ready for V3 model training!")
print("=" * 80)