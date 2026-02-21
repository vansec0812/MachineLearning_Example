import json
import math
import random
from collections import defaultdict

# ======================
# 1. LOAD DATA
# ======================
texts = []
labels = []

def load_data(file_path, label):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            title = item.get("title", "")
            abstract = item.get("abstract", "")
            text = (title + " " + abstract).lower()
            texts.append(text)
            labels.append(label)

load_data('chemistry/chemistry_data.json', 'Chemistry')
load_data('biology/biology_data.json', 'Biology')
load_data('physical/physical_data.json', 'Physical')

# ======================
# 2. TRAIN / TEST SPLIT 70/30
# ======================
random.seed(42)

dataset = list(zip(texts, labels))
random.shuffle(dataset)

split_idx = int(0.7 * len(dataset))
train_data = dataset[:split_idx]

# ======================
# 3. BUILD COUNTS
# ======================
class_word_counts = defaultdict(lambda: defaultdict(int))
class_doc_counts = defaultdict(int)
vocab = set()

for text, label in train_data:
    class_doc_counts[label] += 1
    words = text.split()
    for word in words:
        class_word_counts[label][word] += 1
        vocab.add(word)

vocab_size = len(vocab)
total_docs = len(train_data)

# ======================
# 4. PRIOR P(C)
# ======================
prior_probs = {
    label: class_doc_counts[label] / total_docs
    for label in class_doc_counts
}

# ======================
# 5. LIKELIHOOD P(word | C)
#    (Laplace smoothing)
# ======================
likelihoods = {}

for label in class_word_counts:
    total_words = sum(class_word_counts[label].values())
    likelihoods[label] = {}

    for word in vocab:
        count_word = class_word_counts[label].get(word, 0)
        likelihoods[label][word] = (count_word + 1) / (total_words + vocab_size)

# ======================
# 6. CLASSIFY FUNCTION
# ======================
def classify(text):
    words = text.lower().split()
    scores = {}

    for label in prior_probs:
        score = math.log(prior_probs[label])

        for word in words:
            if word in vocab:
                score += math.log(likelihoods[label][word])

        scores[label] = score

    return max(scores, key=scores.get)

# Test Document
test_doc = """
The interaction between microscopic entities is governed by spatial configuration, 
binding affinity, and environmental constraints. Experimental observations sugge    st 
that changes in surrounding conditions may alter functional outcomes without modifying 
the internal composition. Analytical techniques are applied to examine response patterns 
and transformation efficiency across varying scenarios.
"""
# Chemistry   
# test_doc = 'Chemical reactions are governed by the interaction of atoms and molecules. The study of reaction kinetics focuses on understanding reaction rates and the influence of temperature, catalysts, and concentration. Spectroscopic techniques such as nuclear magnetic resonance and infrared spectroscopy are widely used to analyze molecular structures.'
# Biology 
# test_doc = 'Artificial intelligence technologies simulate human intelligence in robots and computer systems. AI-based models have revolutionized drug development in general over the past decade. System biology is founded on a combination of research that investigates a wide range of dynamic cellular mechanisms while employing mathematical and computational methods that allow for the analysis of varied disease-associated data. Acquiring a precise representation of protein structure is an important initial step in comprehending the principles of biology. While our ability to determine protein structures experimentally has substantially improved due to recent advancements in experimental methods, however, the number of protein sequences and known protein structures differ by a constant amount. Predicting protein structures computationally is one method to close this gap. There have been significant advancements in the field of protein structure prediction recently due to Deep Learning (Deep learning)based approaches as demonstrated by the outcome of AlphaFold2 during the latest Critical Assessment of Protein Structure Prediction (CASP14). The chapter describes various protein structure prediction methods and systems biology-guided smart drug screening'
# Physical
# test_doc = 'Physics explores the fundamental laws of nature, including motion, energy, and forces. Quantum mechanics describes the behavior of particles at the atomic scale, while classical mechanics explains macroscopic phenomena such as gravity and motion. Recent research in particle physics has expanded our understanding of the universe.'
# =========================
prediction = classify(test_doc)

print("VĂN BẢN TEST:")
print(test_doc.strip())
print("\n👉 Dự đoán chủ đề:", prediction)
