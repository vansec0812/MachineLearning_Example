import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


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
            text = text.strip()
            if text != "":
                texts.append(text)
                labels.append(label)

load_data('chemistry/chemistry_data.json', 'Chemistry')
load_data('biology/biology_data.json', 'Biology')
load_data('physical/physical_data.json', 'Physical')

print("Tổng số văn bản:", len(texts))


# ======================
# 2. TRAIN / TEST SPLIT (CHIA TRƯỚC)
# ======================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)


# ======================
# 3. TF-IDF (FIT CHỈ TRÊN TRAIN)
# ======================
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=2000,
    ngram_range=(1, 1)
)

X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)


# ======================
# 4. BASE MODELS
# ======================
nb = MultinomialNB()

svm = CalibratedClassifierCV(
    LinearSVC(random_state=42),
    method='sigmoid'
)

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ======================
# 5. STACKING MODEL
# ======================
stack_model = StackingClassifier(
    estimators=[
        ('nb', nb),
        ('svm', svm),
        ('lr', lr)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)


# ======================
# 6. TRAIN
# ======================
stack_model.fit(X_train, y_train)


# ======================
# 7. EVALUATION (TEST SET)
# ======================
y_pred = stack_model.predict(X_test)

print(f"\nStacking Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ======================
# 8. CONFUSION MATRIX
# ======================
labels_name = ["Biology", "Chemistry", "Physical"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels_name
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels_name,
    yticklabels=labels_name
)

plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Stacking ")
plt.tight_layout()
plt.show()
