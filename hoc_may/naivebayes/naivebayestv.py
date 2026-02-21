import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load dữ liệu từ JSON
# -------------------------------
def load_data():
    datasets = []
    for folder, label in [('biology/biology_data.json', 'Biology'),
                          ('chemistry/chemistry_data.json', 'Chemistry'),
                          ('physical/physical_data.json', 'Physical')]:
        with open(folder, 'r', encoding='utf-8') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            df['label'] = label
            # Tạo cột text = title + abstract
            df['text'] = df['title'] + ' ' + df['abstract']
            datasets.append(df[['text', 'label']])
    return pd.concat(datasets).reset_index(drop=True)

df = load_data()

# -------------------------------
# 2. Đếm text 
# -------------------------------
df = df[df['text'].str.strip() != '']
print(f"Tổng số document sau lọc rỗng: {len(df)}")
counts = df['label'].value_counts().to_dict()
for label, count in counts.items():
    print(f"{label}: {count}")

# -------------------------------
# 3. Tách train/test
# -------------------------------
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -------------------------------
# 4. Vector hóa TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english', max_features=5000, ngram_range=(1, 1)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -------------------------------
# 5. Huấn luyện Naive Bayes
# -------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# -------------------------------
# 6. Dự đoán & đánh giá
# -------------------------------
y_pred = nb_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# -------------------------------
# 7. Hiển thị ma trận nhầm lẫn
# -------------------------------
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=nb_model.classes_, yticklabels=nb_model.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Naive Bayes')
plt.show()
