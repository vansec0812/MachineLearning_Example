import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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
            text = title + " " + abstract

            texts.append(text)
            labels.append(label)

# Load dữ liệu
load_data('chemistry/chemistry_data.json', 'Chemistry')
load_data('biology/biology_data.json', 'Biology')
load_data('physical/physical_data.json', 'Physical')

print("Tổng số văn bản:", len(texts))

# ======================
# 2. TF-IDF VECTORIZE
# ======================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(texts)
y = labels

# ======================
# 3. TRAIN / TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# ======================
# 4. LOGISTIC REGRESSION
# ======================
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)
# tap train
model.fit(X_train, y_train)

# ======================
# 5. EVALUATION
# ======================
y_pred = model.predict(X_test)

# ======================
# 6. DỰ ĐOÁN VĂN BẢN HOÀN CHỈNH
# ======================
test_doc = """
The interaction between microscopic entities is governed by spatial configuration, 
binding affinity, and environmental constraints. Experimental observations suggest 
that changes in surrounding conditions may alter functional outcomes without modifying 
the internal composition. Analytical techniques are applied to examine response patterns 
and transformation efficiency across varying scenarios.
""" 
# Mẫu AI kết hợp Chemistry + Biology
# test_doc = 'The recent advancements in artificial intelligence have significantly impacted the field of biology. Researchers are now able to simulate complex cellular processes using machine learning algorithms, which allows for faster identification of potential drug candidates. However, despite these innovations, traditional chemical experiments remain essential for validating predictions. While the computational models can predict protein folding and interactions, they cannot yet fully replace laboratory-based testing. On the other hand, some chemists argue that relying too heavily on AI might overlook unexpected chemical reactions that are not represented in training datasets. Interestingly, the integration of AI in chemistry and biology often leads to ambiguous results when evaluating multi-disciplinary studies. For instance, predicting the effect of a new compound on human cells requires both a deep understanding of chemical properties and biological pathways. Therefore, although AI-driven approaches offer unprecedented speed, they do not automatically guarantee accuracy or complete insight into the mechanisms involved.'
# Chemistry   
# test_doc = 'Chemical reactions are governed by the interaction of atoms and molecules. The study of reaction kinetics focuses on understanding reaction rates and the influence of temperature, catalysts, and concentration. Spectroscopic techniques such as nuclear magnetic resonance and infrared spectroscopy are widely used to analyze molecular structures.'
# Biology
# test_doc = 'Artificial intelligence technologies simulate human intelligence in robots and computer systems. AI-based models have revolutionized drug development in general over the past decade. System biology is founded on a combination of research that investigates a wide range of dynamic cellular mechanisms while employing mathematical and computational methods that allow for the analysis of varied disease-associated data. Acquiring a precise representation of protein structure is an important initial step in comprehending the principles of biology. While our ability to determine protein structures experimentally has substantially improved due to recent advancements in experimental methods, however, the number of protein sequences and known protein structures differ by a constant amount. Predicting protein structures computationally is one method to close this gap. There have been significant advancements in the field of protein structure prediction recently due to Deep Learning (Deep learning)based approaches as demonstrated by the outcome ofAlphaFold2 during the latest Critical Assessment of Protein Structure Prediction (CASP14). The chapter describes various protein structure prediction methods and systems biology-guided smart drug screening'
# Physical
# test_doc = 'Physics explores the fundamental laws of nature, including motion, energy, and forces. Quantum mechanics describes the behavior of particles at the atomic scale, while classical mechanics explains macroscopic phenomena such as gravity and motion. Recent research in particle physics has expanded our understanding of the universe.'
test_vec = vectorizer.transform([test_doc])
prediction = model.predict(test_vec)
probability = model.predict_proba(test_vec)

print("\nVĂN BẢN TEST:")
print(test_doc.strip())
print("\n👉 Dự đoán chủ đề:", prediction[0])