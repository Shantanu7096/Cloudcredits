# 💉 Predicting Diabetes Using Machine Learning  

## 📌 Project Overview  
This project focuses on predicting whether a **patient has diabetes** using **medical attributes**. It utilizes the **Pima Indians Diabetes Dataset**, which contains several health-related variables recorded for female patients.  

---

## 📂 Dataset Information  

**Dataset:** Pima Indians Diabetes Dataset  
**Target:** Outcome (1 = Diabetic, 0 = Non-Diabetic)  

**Features Include:**  
- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  

---

## 🧠 Algorithms Used  
- **K-Nearest Neighbors (KNN):** Classifies based on closest data points  
- **Logistic Regression:** Predicts binary outcome using linear combinations of features  

---

## 📊 Evaluation Metrics  
- **Accuracy:** Overall prediction correctness  
- **Precision:** Percentage of positive predictions that are correct  
- **Recall:** Percentage of actual diabetics correctly identified  

---

## 🏗️ Project Workflow  

1. **Load Dataset** from CSV  
2. **Clean & Preprocess:** Handle missing values and normalize features  
3. **Train-Test Split:** Typically 80/20 or 70/30  
4. **Model Training:** Train both KNN and Logistic Regression models  
5. **Evaluate Performance:** Use Accuracy, Precision, and Recall  
6. **Visualization:** Optional - Confusion Matrix, ROC curve, etc.  

---

## 🔧 Setup Instructions  

### 1️⃣ Install Dependencies  
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 2️⃣ Run the Script  
```bash
python diabetes_prediction.py
```

- This trains the models and prints evaluation results.  

---

## 🚀 Future Improvements  
- Use **GridSearchCV** to fine-tune KNN `k` values  
- Experiment with **SVM, Random Forest** or **XGBoost**  
- Build a simple web app using **Streamlit** or **Flask**  

---

## 🙌 Credits  
- **Developer:** Shantanu
- **Dataset Source:** UCI Machine Learning Repository  
- **Tools & Libraries:** Python, NumPy, Pandas, Scikit-learn  
