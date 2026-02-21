import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split   # THÊM

# =========================
# LOAD DATA
# =======================
def load_data():
    datasets = []
    for path, label in [
        ('biology/biology_data.json', 'Biology'),
        ('chemistry/chemistry_data.json', 'Chemistry'),
        ('physical/physical_data.json', 'Physics')
    ]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            df['label'] = label
            df['text'] = (df['title'] + ' ' + df['abstract']).str.strip()
            df = df[df['text'] != ""]
            datasets.append(df[['text', 'label']])
    return pd.concat(datasets).reset_index(drop=True)

df = load_data()

print("Total documents:", len(df))

# =========================
# SPLIT DATA 70/30  (THÊM)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.3,
    random_state=42,
    stratify=df["label"]
)

# =========================
# TRAIN SVM
# =========================
vectorizer = TfidfVectorizer(stop_words="english")

# CHỈ ĐỔI NGUỒN FIT SANG TRAIN
X = vectorizer.fit_transform(X_train)
y = y_train
# C là điểm cân bằng hình siêu phẳng
model = LinearSVC(C=1.0, class_weight="balanced")
model.fit(X, y)
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
# PREDICT
# =========================
vec = vectorizer.transform([test_doc])
pred = model.predict(vec)[0]

print("\n===== TEST DOCUMENT =====")
print(test_doc.strip())
print("\nPredicted field:", pred)
