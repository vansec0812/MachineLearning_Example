import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD & CLEAN DATA (Lọc sạch dữ liệu)
# ==========================================
def load_and_clean_data():
    datasets = []
    # Đường dẫn thực tế trên Colab của bạn
    paths = [
        ('biology/biology_data.json', 'Biology'),
        ('chemistry/chemistry_data.json', 'Chemistry'),
        ('physical/physical_data.json', 'Physical')
    ]
    
    for path, label in paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df['label'] = label
                # Gộp Title và Abstract
                df['text'] = (df['title'] + ' ' + df['abstract']).str.strip()
                datasets.append(df[['text', 'label']])
    
    full_df = pd.concat(datasets).reset_index(drop=True)
    # Lọc bỏ các dòng rỗng
    full_df = full_df[full_df['text'] != ""]
    return full_df

df = load_and_clean_data()
print(f"Tổng số văn bản sau khi lọc: {len(df)}")
print(df['label'].value_counts())
X_raw = df['text']
y = df['label']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,  
    ngram_range=(1, 1)
)

# CHỈ HỌC từ vựng từ tập Train
X_train = vectorizer.fit_transform(X_train_raw)
# Áp dụng từ vựng đã học sang tập Test
X_test = vectorizer.transform(X_test_raw)

# ==========================================
# 4. ĐỊNH NGHĨA CÁC MÔ HÌNH
# ==========================================
models = {
    "Naive Bayes": MultinomialNB(),
    
    "SVM": CalibratedClassifierCV(
        LinearSVC(random_state=42), 
        method="sigmoid"
    ),
    
    "Logistic Regression": LogisticRegression(
        max_iter=1000, 
        random_state=42
    )
}

# ==========================================
# 5. HUẤN LUYỆN & SO SÁNH CÔNG BẰNG
# ==========================================
results = {}
print("\n===== KẾT QUẢ SO SÁNH CHI TIẾT =====")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# ==========================================
# 6. STACKING ENSEMBLE (Mô hình kết hợp)
# ==========================================
stack_model = StackingClassifier(
    estimators=[
        ('nb', MultinomialNB()),
        ('svm', CalibratedClassifierCV(LinearSVC(random_state=42), method='sigmoid')),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)

stack_model.fit(X_train, y_train)
y_pred_stack = stack_model.predict(X_test)

stack_acc = accuracy_score(y_test, y_pred_stack)
results["Stacking"] = stack_acc

print("\n--- Stacking ---")
print(f"Accuracy: {stack_acc:.4f}")
print(classification_report(y_test, y_pred_stack))
print("\n===== BẢNG TỔNG KẾT ĐỘ CHÍNH XÁC (ACCURACY) =====")
# Tạo DataFrame từ dictionary kết quả
summary_table = pd.DataFrame(list(results.items()), columns=['Mô hình', 'Accuracy'])
# Sắp xếp giảm dần theo Accuracy
summary_table = summary_table.sort_values(by='Accuracy', ascending=False)
# Cách in ra với định dạng 4 chữ số thập phân
print(summary_table.to_string(index=False, formatters={'Accuracy': '{:,.4f}'.format}))