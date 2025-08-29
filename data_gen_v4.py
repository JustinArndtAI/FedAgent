#!/usr/bin/env python
"""
V4 DATA GENERATION - 50K+ APOCALYPSE-LEVEL SAMPLES
FINAL BOSS MODE: OBLITERATE ALL TARGETS
"""
import numpy as np
import random
import nltk
from nltk.corpus import movie_reviews, reuters, brown, gutenberg, webtext, nps_chat
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("V4 DATA APOCALYPSE - 50K+ HYPER-REAL SAMPLES")
print("=" * 80)

# Download all NLTK data
print("\nDownloading NLTK corpora...")
for corpus in ['movie_reviews', 'reuters', 'brown', 'gutenberg', 'webtext', 'nps_chat', 'vader_lexicon']:
    try:
        nltk.download(corpus, quiet=True)
    except:
        pass

# ALIGNMENT DATA: 50K samples from multiple corpora
print("\n[1/2] Generating 50K Alignment Dataset...")

alignment_texts = []
alignment_labels = []

# Professional therapy responses (score: 0.9-1.0)
professional_templates = [
    "I understand {concern}. Your feelings are completely valid and I'm here to support you.",
    "Thank you for trusting me with {concern}. Let's explore what would be most helpful.",
    "I hear {concern} in your words, and I want to acknowledge how difficult this must be.",
    "It takes courage to share {concern}. You're not alone in this journey.",
    "Your wellbeing regarding {concern} is my priority. Let's work together.",
    "I appreciate your openness about {concern}. This is a safe space.",
    "What you're experiencing with {concern} is more common than you might think.",
    "I can see the strength it takes to discuss {concern}. That's an important step.",
    "Your perspective on {concern} matters, and I'm here to listen without judgment.",
    "Let's take {concern} one step at a time. There's no pressure.",
]

concerns = [
    "this difficult situation", "your anxiety", "these feelings", "this challenge",
    "your depression", "this trauma", "your fears", "this pain", "your struggles",
    "this loss", "your stress", "this change", "your relationship", "this crisis"
]

# Generate 15K professional responses with variations
print("  Generating professional therapy responses...")
for _ in range(15000):
    template = random.choice(professional_templates)
    concern = random.choice(concerns)
    text = template.format(concern=concern)
    
    # Add contextual variations
    if random.random() > 0.5:
        additions = [
            " I believe in your resilience.",
            " You have the strength to overcome this.",
            " Together, we can work through this.",
            " Your mental health is important.",
            " I'm committed to supporting you.",
        ]
        text += random.choice(additions)
    
    alignment_texts.append(text)
    alignment_labels.append(random.uniform(0.9, 1.0))  # High alignment

# Poor/harmful responses (score: 0.0-0.3)
print("  Generating poor/harmful responses...")
poor_templates = [
    "Just get over {problem} already.",
    "Stop being so dramatic about {problem}.",
    "{problem} isn't a real issue.",
    "You're overreacting to {problem}.",
    "Other people have worse {problem} than you.",
    "Man up and deal with {problem}.",
    "{problem} is your own fault.",
    "Nobody cares about your {problem}.",
    "You're weak for having {problem}.",
    "Stop seeking attention with {problem}.",
]

problems = ["it", "this", "that", "your issues", "problems", "feelings", "emotions"]

for _ in range(10000):
    template = random.choice(poor_templates)
    problem = random.choice(problems)
    text = template.format(problem=problem)
    alignment_texts.append(text)
    alignment_labels.append(random.uniform(0.0, 0.3))  # Low alignment

# Extract from Brown corpus for neutral/mixed (score: 0.4-0.7)
print("  Extracting from Brown corpus...")
try:
    brown_files = brown.fileids()[:5000]
    for fid in brown_files:
        try:
            text = ' '.join(brown.words(fid)[:50])  # First 50 words
            alignment_texts.append(text)
            alignment_labels.append(random.uniform(0.4, 0.7))
        except:
            continue
except:
    print("    Brown corpus not available, using synthetic data")
    for _ in range(5000):
        text = f"This is a neutral statement about {random.choice(['therapy', 'counseling', 'support'])}."
        alignment_texts.append(text)
        alignment_labels.append(random.uniform(0.4, 0.7))

# Extract from Reuters for informational (score: 0.5-0.8)
print("  Extracting from Reuters corpus...")
try:
    reuters_files = reuters.fileids()[:5000]
    for fid in reuters_files:
        try:
            text = ' '.join(reuters.words(fid)[:50])
            alignment_texts.append(text)
            alignment_labels.append(random.uniform(0.5, 0.8))
        except:
            continue
except:
    print("    Reuters corpus not available, using synthetic data")
    for _ in range(5000):
        text = f"Information about {random.choice(['mental health', 'wellbeing', 'therapy'])} services."
        alignment_texts.append(text)
        alignment_labels.append(random.uniform(0.5, 0.8))

# Generate empathetic responses from templates (score: 0.8-0.95)
print("  Generating empathetic variations...")
empathy_templates = [
    "I can see how {feeling} you must be feeling.",
    "It sounds like you're really {feeling} right now.",
    "I imagine this makes you feel {feeling}.",
    "Many people feel {feeling} in similar situations.",
    "Feeling {feeling} is a natural response.",
]

feelings = ["anxious", "overwhelmed", "scared", "frustrated", "sad", "confused", 
           "angry", "lonely", "helpless", "exhausted"]

for _ in range(10000):
    template = random.choice(empathy_templates)
    feeling = random.choice(feelings)
    text = template.format(feeling=feeling)
    alignment_texts.append(text)
    alignment_labels.append(random.uniform(0.8, 0.95))

# Ensure we have exactly 50K samples
while len(alignment_texts) < 50000:
    # Generate synthetic professional responses
    text = f"I understand you're experiencing {random.choice(concerns)}. {random.choice(['Your feelings are valid.', 'I am here to help.', 'Let us work through this together.'])}"
    alignment_texts.append(text)
    alignment_labels.append(random.uniform(0.85, 1.0))

# Trim to exactly 50K
alignment_texts = alignment_texts[:50000]
alignment_labels = alignment_labels[:50000]

print(f"✓ Generated {len(alignment_texts)} alignment samples")

# Save alignment data
np.save("v4_align_texts.npy", np.array(alignment_texts, dtype=object))
np.save("v4_align_labels.npy", np.array(alignment_labels))

# WELLBEING DATA: 50K samples
print("\n[2/2] Generating 50K Wellbeing Dataset...")

wellbeing_texts = []
wellbeing_scores = []

# Crisis texts (score: -1.0 to -0.8)
crisis_keywords = [
    "want to end my life", "kill myself", "can't go on", "no point in living",
    "better off dead", "planning to die", "want to disappear", "end it all",
    "hurt myself", "give up on life", "no hope left", "too much pain"
]

print("  Generating crisis texts...")
for _ in range(5000):
    base = random.choice(crisis_keywords)
    text = f"I {base} because {random.choice(['everything hurts', 'life is meaningless', 'nobody cares', 'I am worthless'])}"
    wellbeing_texts.append(text)
    wellbeing_scores.append(random.uniform(-1.0, -0.8))

# Depression texts (score: -0.7 to -0.4)
depression_templates = [
    "I feel so {adjective} and {emotion}",
    "Everything is {adjective}, I can't {action}",
    "I'm {emotion} all the time and {adjective}",
    "Life feels {adjective} and I'm {emotion}",
]

dep_adjectives = ["hopeless", "empty", "worthless", "broken", "lost", "defeated"]
dep_emotions = ["depressed", "sad", "miserable", "numb", "exhausted", "alone"]
dep_actions = ["function", "cope", "continue", "manage", "handle it", "go on"]

print("  Generating depression texts...")
for _ in range(10000):
    template = random.choice(depression_templates)
    text = template.format(
        adjective=random.choice(dep_adjectives),
        emotion=random.choice(dep_emotions),
        action=random.choice(dep_actions)
    )
    wellbeing_texts.append(text)
    wellbeing_scores.append(random.uniform(-0.7, -0.4))

# Anxiety texts (score: -0.5 to -0.2)
print("  Generating anxiety texts...")
anxiety_templates = [
    "I'm so anxious about {worry}",
    "I can't stop worrying about {worry}",
    "My anxiety about {worry} is overwhelming",
    "I'm panicking about {worry}",
]

worries = ["everything", "the future", "work", "relationships", "my health", 
          "money", "failing", "being judged", "making mistakes"]

for _ in range(10000):
    template = random.choice(anxiety_templates)
    text = template.format(worry=random.choice(worries))
    wellbeing_texts.append(text)
    wellbeing_scores.append(random.uniform(-0.5, -0.2))

# Neutral texts (score: -0.2 to 0.2)
print("  Generating neutral texts...")
neutral_statements = [
    "Today is just another day",
    "Things are okay I guess",
    "Nothing special happening",
    "Life goes on as usual",
    "Same routine as always",
    "Just getting by",
    "It's an average day",
    "Nothing to report",
]

for _ in range(10000):
    text = random.choice(neutral_statements)
    if random.random() > 0.5:
        text += f", {random.choice(['nothing more', 'nothing less', 'as expected', 'like always'])}"
    wellbeing_texts.append(text)
    wellbeing_scores.append(random.uniform(-0.2, 0.2))

# Positive texts (score: 0.3 to 0.7)
print("  Generating positive texts...")
positive_templates = [
    "I'm feeling {good} about {aspect}",
    "Things are {good} with {aspect}",
    "{aspect} is going {good}",
    "I'm {good} today because of {aspect}",
]

good_words = ["good", "better", "positive", "hopeful", "optimistic", "content"]
aspects = ["life", "work", "my progress", "relationships", "my health", "the future"]

for _ in range(10000):
    template = random.choice(positive_templates)
    text = template.format(
        good=random.choice(good_words),
        aspect=random.choice(aspects)
    )
    wellbeing_texts.append(text)
    wellbeing_scores.append(random.uniform(0.3, 0.7))

# Euphoric texts (score: 0.8 to 1.0)
print("  Generating euphoric texts...")
euphoric_expressions = [
    "I'm absolutely thrilled with life!",
    "Everything is amazing and wonderful!",
    "I've never been happier!",
    "Life is absolutely perfect!",
    "I'm on top of the world!",
    "This is the best day ever!",
    "I'm bursting with joy and happiness!",
    "Everything is going perfectly!",
    "I feel invincible and unstoppable!",
    "Life couldn't possibly be better!",
]

for _ in range(5000):
    text = random.choice(euphoric_expressions)
    if random.random() > 0.5:
        text += f" {random.choice(['Amazing!', 'Incredible!', 'Fantastic!', 'Wonderful!'])}"
    wellbeing_texts.append(text)
    wellbeing_scores.append(random.uniform(0.8, 1.0))

# Ensure exactly 50K samples
while len(wellbeing_texts) < 50000:
    # Add more neutral/mixed
    text = f"Feeling {random.choice(['okay', 'alright', 'fine', 'normal'])} today"
    wellbeing_texts.append(text)
    wellbeing_scores.append(random.uniform(-0.1, 0.3))

wellbeing_texts = wellbeing_texts[:50000]
wellbeing_scores = wellbeing_scores[:50000]

print(f"✓ Generated {len(wellbeing_texts)} wellbeing samples")

# Save wellbeing data
with open("v4_wellbeing_texts.txt", "w", encoding="utf-8") as f:
    for text in wellbeing_texts:
        f.write(text + "\n")

np.save("v4_wellbeing_scores.npy", np.array(wellbeing_scores))

# Generate statistics
print("\n" + "=" * 80)
print("V4 DATA GENERATION COMPLETE - APOCALYPSE ACHIEVED")
print("=" * 80)
print(f"\nDataset Statistics:")
print(f"  - Total Alignment Samples: {len(alignment_texts)}")
print(f"  - Total Wellbeing Samples: {len(wellbeing_texts)}")
print(f"  - Combined Dataset Size: {len(alignment_texts) + len(wellbeing_texts)}")
print(f"\nAlignment Distribution:")
print(f"  - High (0.9-1.0): {sum(1 for x in alignment_labels if x >= 0.9)}")
print(f"  - Good (0.7-0.9): {sum(1 for x in alignment_labels if 0.7 <= x < 0.9)}")
print(f"  - Medium (0.4-0.7): {sum(1 for x in alignment_labels if 0.4 <= x < 0.7)}")
print(f"  - Poor (0.0-0.4): {sum(1 for x in alignment_labels if x < 0.4)}")
print(f"\nWellbeing Distribution:")
print(f"  - Crisis (<-0.8): {sum(1 for x in wellbeing_scores if x < -0.8)}")
print(f"  - Depression (-0.8 to -0.4): {sum(1 for x in wellbeing_scores if -0.8 <= x < -0.4)}")
print(f"  - Anxiety (-0.4 to -0.2): {sum(1 for x in wellbeing_scores if -0.4 <= x < -0.2)}")
print(f"  - Neutral (-0.2 to 0.2): {sum(1 for x in wellbeing_scores if -0.2 <= x <= 0.2)}")
print(f"  - Positive (0.2 to 0.8): {sum(1 for x in wellbeing_scores if 0.2 < x <= 0.8)}")
print(f"  - Euphoric (>0.8): {sum(1 for x in wellbeing_scores if x > 0.8)}")
print("\n✓ Files created:")
print("  - v4_align_texts.npy (50K samples)")
print("  - v4_align_labels.npy")
print("  - v4_wellbeing_texts.txt (50K samples)")
print("  - v4_wellbeing_scores.npy")
print("\n🔥 READY FOR V4 FINAL BOSS MODEL TRAINING")
print("=" * 80)