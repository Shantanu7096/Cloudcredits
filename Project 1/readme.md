# 🏡 Predicting House Prices  

## 📌 Project Overview  

This project aims to predict house prices based on various features like **size, location, and number of rooms** using **Linear Regression**. The dataset used is the **Boston Housing Dataset**, which contains multiple attributes influencing house prices.  

---

## 📁 Dataset Information  

**Dataset:** Boston Housing Dataset  
**Features:**  
- CRIM: Crime rate per town  
- ZN: Proportion of residential land zoned for large lots  
- INDUS: Proportion of non-retail business acres per town  
- CHAS: Charles River dummy variable  
- NOX: Nitrogen oxide concentration  
- RM: Average number of rooms per dwelling  
- AGE: Proportion of owner-occupied units built before 1940  
- DIS: Weighted distances to employment centers  
- RAD: Accessibility to radial highways  
- TAX: Property tax rate  
- PTRATIO: Pupil-teacher ratio  
- B: Proportion of Black residents  
- LSTAT: Lower income population percentage  
- **MEDV:** Median house value (Target Variable)  

---

## 🏗️ Project Steps  

### 📊 Data Preprocessing  
1. **Load Dataset** from CSV file.  
2. **Remove Target Variable** (`MEDV`) from features.  
3. **Split Data** into training and testing sets (80% train, 20% test).  
4. **Normalize Features** (if needed for better model performance).  

### 🤖 Model Training  
- Implement **Linear Regression**.  
- Fit the model using training data.  

### 📈 Model Evaluation  
- **Mean Squared Error (MSE):** Measures average squared difference between actual and predicted prices.  
- **R² Score:** Determines variance explained by the model.  

### 🖼️ Visualization  
- **Bar Graph:** Compares actual vs predicted house prices for better understanding.  

---

## 🔧 Setup Instructions  

### 1️⃣ Install Dependencies  
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 2️⃣ Run the Script  
```bash
python house_price_prediction.py
```

- This will train the model and output evaluation metrics.  
- The final graph will show **actual vs predicted house prices**.  

---
### Screenshot - Output

![image](https://github.com/user-attachments/assets/0a9d775c-4a00-45e6-8028-02355d1c705f)

```
```
## 🚀 Future Improvements  

- Experiment with **Ridge Regression** or **Lasso Regression** for better regularization.  
- Perform **feature selection** to improve model accuracy.  
- Use **Deep Learning models** for complex price predictions.  

---

## 🙌 Credits  

- **Developer:**Shantanu
- **Dataset:** Boston Housing Dataset  
- **Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib  
