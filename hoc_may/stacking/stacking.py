import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split

# ======================
# 1. LOAD DATA
# ======================
texts = []
labels = []

def load_data(file_path, label):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            text = item.get("title", "") + " " + item.get("abstract", "")
            texts.append(text)
            labels.append(label)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_data(os.path.join(BASE_DIR, 'chemistry', 'chemistry_data.json'), 'Chemistry')
load_data(os.path.join(BASE_DIR, 'biology', 'biology_data.json'), 'Biology')
load_data(os.path.join(BASE_DIR, 'physical', 'physical_data.json'), 'Physical')

# ======================
# 2. TF-IDF
# ======================
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 1)
)

X = vectorizer.fit_transform(texts)
y = labels

# ======================
# 2.1 SPLIT DATA (70% train, 30% test)
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ======================
# 3. STACKING MODEL
# ======================
nb = MultinomialNB()

svm = CalibratedClassifierCV(
    LinearSVC(),
    method='sigmoid'
)

lr = LogisticRegression(
    max_iter=1000
)

stack_model = StackingClassifier(
    estimators=[
        ('nb', nb),
        ('svm', svm),
        ('lr', lr)
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

# ======================
# 4. TRAIN (on 70% data)
# ======================
stack_model.fit(X_train, y_train)

# ======================
# 5. PREDICT NEW DOCUMENT
# ======================
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

test_vec = vectorizer.transform([test_doc])
prediction = stack_model.predict(test_vec)
probability = stack_model.predict_proba(test_vec)

print("\n📄 VĂN BẢN TEST:")
print(test_doc.strip())
print("\n👉 Dự đoán chủ đề:", prediction[0])
# print("\n👉 Xác suất:", probability)
