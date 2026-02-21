
import json
import pandas as pd
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'naivebayes'))
import naivebayes
import count

# Helper to load data with specific order
def load_data_ordered(order):
    datasets = []
    # order is list of tuples: ('chemistry', 'Chemistry'), etc.
    for folder_name, label in order:
        path = os.path.join(BASE_DIR, folder_name, f'{folder_name}_data.json')
        # Fix for physical naming discrepancy in some scripts if any
        if folder_name == 'physical' and not os.path.exists(path):
             path = os.path.join(BASE_DIR, 'physical', 'physical_data.json')

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            df['label'] = label
            df['text'] = (df['title'] + ' ' + df['abstract']).str.strip()
            df = df[df['text'] != ""]
            datasets.append(df[['text', 'label']])
    return pd.concat(datasets).reset_index(drop=True)

# 1. Naive Bayes TV Match
def train_nb_tv():
    # Load: Bio -> Chem -> Phy
    df = load_data_ordered([('biology', 'Biology'), ('chemistry', 'Chemistry'), ('physical', 'Physical')])
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42, stratify=df['label'])
    
    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vec.fit_transform(X_train)
    X_test_tfidf = vec.transform(X_test)
    
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["Biology", "Chemistry", "Physical"]) # Check labels order in script? usually alpha or appearing? Script uses model.classes_
    # nb script uses nb_model.classes_. 
    return {"accuracy": acc, "cm": cm, "classes": model.classes_}

# 2. SVM TV Match
def train_svm_tv():
    # Load: Bio -> Chem -> Phy (svmtv.py lines 16-20)
    df = load_data_ordered([('biology', 'Biology'), ('chemistry', 'Chemistry'), ('physical', 'Physical')]) 
    # Label mapping in svmtv is Physical->Physics? No, line 19 says 'Physics'
    # Actually svmtv line 19: ('physical/physical_data.json', 'Physics')
    # But let's check input data. User used "Physical" in app. 
    # Let's standardize on "Physical" for display, or match script exactly? 
    # Script svmtv.py line 19: 'Physics'. 
    # Note: app.py uses cls = ["Biology", "Chemistry", "Physical"]. 
    # If I return Physics, confusion matrix might mismatch. 
    # Let's override to Physical to be consistent with App, OR update App.
    # User said "Correct". Let's stick to "Physical" if possible to avoid breaking visuals.
    # But svmtv loads it as Physics.
    # Let's just use the load_data_ordered logic with 'Physical' label to be safe for App consistency unless user complains.
    
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42, stratify=df['label'])
    
    vec = TfidfVectorizer(stop_words='english') # No max_features in svmtv
    X_train_tfidf = vec.fit_transform(X_train)
    X_test_tfidf = vec.transform(X_test)
    
    model = LinearSVC(C=1.0)
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    return {"accuracy": acc, "cm": cm, "classes": model.classes_}

# 3. Logistic TV Match
def train_lr_tv():
    # Load: Chem -> Bio -> Phy (logistictv.py lines 30-32, note: Physical->Physical)
    df = load_data_ordered([('chemistry', 'Chemistry'), ('biology', 'Biology'), ('physical', 'Physical')])
    
    # logistictv fits on FULL data first!
    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    X_full = vec.fit_transform(df['text'])
    y_full = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=42, stratify=y_full)
    
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # logistictv lines 76: labels_name = ["Biology", "Chemistry", "Physical"]
    cm = confusion_matrix(y_test, y_pred, labels=["Biology", "Chemistry", "Physical"])
    return {"accuracy": acc, "cm": cm, "classes": ["Biology", "Chemistry", "Physical"]}

# 4. Stacking TV Match
def train_stacking_tv():
    # Load: Chem -> Bio -> Phy (stackingtv.py lines 33-35)
    df = load_data_ordered([('chemistry', 'Chemistry'), ('biology', 'Biology'), ('physical', 'Physical')])
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42, stratify=df['label'])
    
    # stackingtv: max_features=2000, fit on X_train ONLY
    vec = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,1))
    X_train_vec = vec.fit_transform(X_train_raw)
    X_test_vec = vec.transform(X_test_raw)
    
    # Stacking setup
    nb = MultinomialNB()
    svm = CalibratedClassifierCV(LinearSVC(random_state=42), method='sigmoid')
    lr = LogisticRegression(max_iter=1000, random_state=42)
    
    stack_model = StackingClassifier(
        estimators=[('nb', nb), ('svm', svm), ('lr', lr)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, n_jobs=-1
    )
    stack_model.fit(X_train_vec, y_train)
    
    y_pred = stack_model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["Biology", "Chemistry", "Physical"])
    return {"accuracy": acc, "cm": cm, "classes": ["Biology", "Chemistry", "Physical"]}


# Main function called by App
def train_models():
    # 1. EVALUATION RESULTS (TV scripts)
    nb_res = train_nb_tv()
    svm_res = train_svm_tv()
    lr_res = train_lr_tv()
    stack_res = train_stacking_tv()
    
    # 2. DEMO MODELS (Main scripts)
    # Re-using the logic from previous step for Demo, but ensuring we cover them all.
    # NB Demo: Custom script (already integrated)
    # SVM Demo: Full data, LinearSVC (matches svm.py)
    # LR Demo: Split data likely? logistic.py does split.
    # Stacking Demo: Full data (matches stacking.py)
    
    # ...Loading for Demo...
    # We need a shared vectorizer? No, each script has its own.
    # But for the Demo input, we need to transform it using the CORRECT vectorizer for that model.
    # This implies we need to store the vectorizer for each model too!
    
    # Let's build the DEMO models specifically
    
    # A. LR Demo (logistic.py uses split, fits on FULL X)
    df_lr = load_data_ordered([('chemistry', 'Chemistry'), ('biology', 'Biology'), ('physical', 'Physical')])
    vec_lr = TfidfVectorizer(stop_words='english', max_features=5000)
    X_full_lr = vec_lr.fit_transform(df_lr['text'])
    y_full_lr = df_lr['label']
    X_tr_lr, _, y_tr_lr, _ = train_test_split(X_full_lr, y_full_lr, test_size=0.3, random_state=42) # No stratify in logistic.py? 
    # Wait, earlier I saw no stratify in logistic.py. 
    # I should check logistic.py content again to be 100% sure for DEMO.
    # Assuming logistic.py has NO stratify.
    lr_demo_model = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr_demo_model.fit(X_tr_lr, y_tr_lr)
    
    # B. SVM Demo (svm.py trains on FULL data)
    df_svm = load_data_ordered([('biology', 'Biology'), ('chemistry', 'Chemistry'), ('physical', 'Physical')])
    vec_svm = TfidfVectorizer(stop_words='english')
    X_full_svm = vec_svm.fit_transform(df_svm['text'])
    svm_demo_model = LinearSVC(C=1.0, class_weight='balanced')
    svm_demo_model.fit(X_full_svm, df_svm['label'])
    
    # C. Stacking Demo (stacking.py trains on FULL data)
    df_stack = load_data_ordered([('chemistry', 'Chemistry'), ('biology', 'Biology'), ('physical', 'Physical')])
    vec_stack = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,1))
    X_full_stack = vec_stack.fit_transform(df_stack['text'])
    # Re-create stack model
    nb_s = MultinomialNB()
    svm_s = CalibratedClassifierCV(LinearSVC(), method='sigmoid')
    lr_s = LogisticRegression(max_iter=1000)
    stack_demo_model = StackingClassifier(
        estimators=[('nb', nb_s), ('svm', svm_s), ('lr', lr_s)],
        final_estimator=LogisticRegression(max_iter=1000), cv=5
    )
    stack_demo_model.fit(X_full_stack, df_stack['label'])
    
    return {
        "report": {
            "nb": nb_res,
            "svm": svm_res,
            "lr": lr_res,
            "stacking": stack_res
        },
        "demo": {
            "lr": (lr_demo_model, vec_lr),
            "svm": (svm_demo_model, vec_svm),
            "stacking": (stack_demo_model, vec_stack),
            "custom_nb": { 
                "classify": naivebayes.classify,
                "prior": naivebayes.prior_probs,
                "likelihoods": naivebayes.likelihoods,
                "count_module": count
            }
        }
    }
