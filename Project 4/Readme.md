# 🚢 Predicting Titanic Survivors  

## 📌 Project Overview  

This project aims to predict whether a **passenger survived the Titanic disaster** based on features like **age, gender, and class**. We use **Logistic Regression** and **Random Forest** models for classification.  

---

## 📂 Dataset Information  

**Dataset:** Titanic Dataset  
**Target Variable:** Survived (1 = Yes, 0 = No)  

**Features Used:**  
- `Pclass` – Passenger class (1st, 2nd, 3rd)  
- `Sex` – Gender (Male, Female)  
- `Age` – Age in years  
- `SibSp` – Number of siblings/spouses aboard  
- `Parch` – Number of parents/children aboard  
- `Fare` – Ticket fare  
- `Embarked` – Port of embarkation  

---

## 🏗️ Project Steps  

### 📊 Data Preprocessing  
1. **Handle Missing Values** in `Age` & `Embarked` features.  
2. **Convert Categorical Data** (Sex, Embarked) to numerical form.  
3. **Normalize Features** if necessary.  
4. **Split Data** into training and testing sets (80% train, 20% test).  

### 🤖 Model Training  
- Implement **Logistic Regression** for probabilistic classification.  
- Implement **Random Forest** for better accuracy and robustness.  

### 📈 Model Evaluation  
- **Accuracy:** Measures correct predictions.  
- **Precision:** Measures positive predictions correctly classified.  
- **Recall:** Measures how well survivors are identified.  

### 🖼️ Visualization  
- **Confusion Matrix:** Shows correct vs incorrect classifications.  
- **Feature Importance Plot:** Displays which features impact survival predictions.  

---

## 🔧 Setup Instructions  

### 1️⃣ Install Dependencies  
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2️⃣ Run the Script  
```bash
python titanic_survival_prediction.py
```

- This will train the model and display evaluation metrics.  

---

## Screenshot - Output

![image](https://github.com/user-attachments/assets/75ea10cd-1a68-4ad0-a33b-3e262f8b3acf)



## 🚀 Future Improvements  

- Test with **Gradient Boosting** or **XGBoost** models.  
- Add more features such as **family connections** for better insights.  
- Deploy using **Flask or Streamlit** for live predictions.  

---

## 🙌 Credits  

- **Developer:** Shantanu 
- **Dataset:** Titanic Disaster Dataset  
- **Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib  
