import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# =========================
# LOAD DATA 
# =========================
def load_data():
    datasets = []
    for folder, label in [
        ('biology/biology_data.json', 'Biology'),
        ('chemistry/chemistry_data.json', 'Chemistry'),
        ('physical/physical_data.json', 'Physics')
    ]:
        with open(folder, 'r', encoding='utf-8') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            df['label'] = label
            df['text'] = (df['title'] + ' ' + df['abstract']).str.strip()
            df = df[df['text'] != ""]
            datasets.append(df[['text', 'label']])
    return pd.concat(datasets).reset_index(drop=True)

df = load_data()

# =========================
# THỐNG KÊ 
# =========================
print("Documents per class:")
counts = df["label"].value_counts().to_dict()
for label, count in counts.items():
    print(f"{label}: {count}")

print("Total documents:", len(df))

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.3,
    random_state=42,
    stratify=df["label"]
)

# =========================
# TF-IDF 
# =========================
vectorizer = TfidfVectorizer(
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =========================
# TRAIN SVM
# =========================
model = LinearSVC(C=1.0)
model.fit(X_train_tfidf, y_train)

# =========================
# PREDICT
# =========================
y_pred = model.predict(X_test_tfidf)

# =========================
# EVALUATION
# =========================
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX 
# =========================
cm = confusion_matrix(y_test, y_pred)
labels = model.classes_

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - SVM")
plt.colorbar()

plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center", color="black")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
