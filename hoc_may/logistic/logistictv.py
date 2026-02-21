import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

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

load_data('chemistry/chemistry_data.json', 'Chemistry')
load_data('biology/biology_data.json', 'Biology')
load_data('physical/physical_data.json', 'Physical')

print("Tổng số văn bản:", len(texts))

# ======================
# 2. TF-IDF
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
    random_state=42,
    stratify=y
)

# ======================
# 4. LOGISTIC REGRESSION
# ======================
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)
# huấn luyện trên tập train
model.fit(X_train, y_train)
#  dự đoán trên tập test
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ======================
# 5. CONFUSION MATRIX
# ======================
labels_name = ["Biology", "Chemistry", "Physical"]
cm = confusion_matrix(y_test, y_pred, labels=labels_name)

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
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()
