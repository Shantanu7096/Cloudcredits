# 🧪 Breast Cancer Prediction  

## 📌 Project Overview  

This project aims to **predict whether a tumor is malignant or benign** using classification algorithms. By leveraging the **Breast Cancer Wisconsin Dataset**, the model helps in the early detection of breast cancer through machine learning techniques.  

---

## 📂 Dataset Information  

**Dataset:** Breast Cancer Wisconsin Dataset  
**Target Variable:**  
- 0: Benign  
- 1: Malignant  

**Features Include:**  
- Radius (mean)  
- Texture (mean)  
- Perimeter (mean)  
- Area (mean)  
- Smoothness (mean)  
- ...and other relevant computed features  

This dataset contains **569 instances** and **30 numerical features**.

---

## 🧠 Algorithms Used  

- **Support Vector Machine (SVM):** Finds the optimal hyperplane to separate malignant and benign classes  
- **Random Forest:** Ensemble of decision trees for robust classification  

---

## 📊 Evaluation Metrics  

- **Accuracy:** Overall correctness of predictions  
- **Precision:** Proportion of true positives among all predicted positives  
- **Recall:** Proportion of true positives correctly identified  

---

## 🏗️ Project Workflow  

1. **Load Dataset** from scikit-learn or CSV  
2. **Data Preprocessing:**  
   - Check for missing values  
   - Normalize features (if needed)  
3. **Train-Test Split**  
4. **Model Building:** Train SVM and Random Forest classifiers  
5. **Evaluate Models:** Generate accuracy, precision, recall  
6. **Visualizations:**  
   - Confusion Matrix  
   - Feature Importance (for Random Forest)  

---

## 🔧 Setup Instructions  

### 1️⃣ Install Dependencies  
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2️⃣ Run the Script  
```bash
python breast_cancer_prediction.py
```
---
## Screenshot - Output

![image](https://github.com/user-attachments/assets/b412d01b-735f-4149-a0be-33e013f3bd45)

---

## 🚀 Future Enhancements  

- Try additional algorithms like **XGBoost** or **Neural Networks**  
- Tune hyperparameters using **GridSearchCV**  
- Build an interactive tool using **Streamlit** or **Flask**  

---

## 🙌 Credits  

- **Developer:** Shantanu 
- **Dataset Source:** UCI Machine Learning Repository  
- **Tech Stack:** Python, Scikit-learn, Matplotlib, Seaborn 
