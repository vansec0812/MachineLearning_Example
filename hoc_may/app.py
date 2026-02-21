import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import model_utils
from sklearn.metrics import accuracy_score, confusion_matrix

# Caching data and models to avoid retraining on every reload
@st.cache_resource
def load_and_train_v5():
    return model_utils.train_models()

data = load_and_train_v5()

st.title("🎓 Báo cáo Phân loại Văn bản Học máy")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Giới thiệu", "So sánh Kết quả", "Thử nghiệm Kết quả"])

cls = ["Biology", "Chemistry", "Physical"]

# --------------------
# TAB 1: GIỚI THIỆU
# --------------------
with tab1:
    st.header("Tổng quan dự án")
    st.markdown("")
    st.write("Dự án xây dựng các mô hình machine learning để phân loại các văn bản khoa học về các lĩnh vực:")
    st.markdown("""
    * **Biology** (Sinh học)
    * **Chemistry** (Hóa học)
    * **Physical** (Vật lý)
    """)
    st.write("Các mô hình được sử dụng:")
    st.markdown("""
    1. **Naive Bayes** 
    2. **Support Vector Machine (SVM)**
    3. **Logistic Regression**
    4. **Stacking Ensemble** (Kết hợp Naive Bayes, SVM, và Logistic Regression)
    """)

# --------------------
# TAB 2: SO SÁNH KẾT QUẢ
# --------------------
with tab2:
    st.header("Kết quả Độ chính xác")
    report_data = data["report"]
    results = [
        {"Model": "Naive Bayes", "Accuracy": report_data["nb"]["accuracy"]},
        {"Model": "SVM", "Accuracy": report_data["svm"]["accuracy"]},
        {"Model": "Logistic Regression", "Accuracy": report_data["lr"]["accuracy"]},
        {"Model": "Stacking", "Accuracy": report_data["stacking"]["accuracy"]},
    ]

    res_df = pd.DataFrame(results)
    st.dataframe(res_df)

    st.bar_chart(res_df.set_index("Model"))

    st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
    selected_model_cm = st.selectbox("Chọn mô hình:",
                                     ["Naive Bayes", "SVM", "Logistic Regression", "Stacking"])
    
    key_map = {
        "Naive Bayes": "nb",
        "SVM": "svm",
        "Logistic Regression": "lr",
        "Stacking": "stacking"
    }
    
    selected_key = key_map[selected_model_cm]
    cm_data = report_data[selected_key]
    cm = cm_data["cm"]
    classes = cm_data["classes"] # labeling

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# --------------------
# TAB 3: THỬ NGHIỆM KẾT QUẢ
# --------------------
with tab3:
    st.header("Dự đoán văn bản mới")
    st.write("Nhập tiêu đề và tóm tắt bài báo để dự đoán lĩnh vực:")
    text_input = st.text_area("Văn bản (Title + Abstract):", height=150)

    if st.button("Dự đoán"):
        if text_input.strip() == "":
            st.warning("Vui lòng nhập nội dung văn bản.")
        else:
            # Use DEMO models
            demo = data["demo"]
            
            # 1. Custom NB Prediction
            cnb = demo["custom_nb"]
            filter_words = cnb["count_module"].convert_todict(text_input)
            test_doc_list = filter_words.keys()
            nb_pred = cnb["classify"](test_doc_list)

            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**Naive Bayes**")
                st.info(nb_pred) 
            
            with col2:
                st.markdown("**SVM**")
                model, vec = demo["svm"]
                pred = model.predict(vec.transform([text_input]))[0]
                st.info(pred)
            
            with col3:
                st.markdown("**Logistic Regression**")
                model, vec = demo["lr"]
                pred = model.predict(vec.transform([text_input]))[0]
                st.info(pred)
            
            with col4:
                st.markdown("**Stacking**")
                model, vec = demo["stacking"]
                pred = model.predict(vec.transform([text_input]))[0]
                st.success(pred)
