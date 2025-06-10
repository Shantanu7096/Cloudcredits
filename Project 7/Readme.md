# 📧 Spam Email Detection  

## 📌 Project Overview  

This project aims to **classify emails as spam or not spam** using classic machine learning techniques. We apply **Naive Bayes** and **Support Vector Machine (SVM)** algorithms to the **Enron Email Dataset** to train a text classifier capable of filtering out unwanted messages.  

---

## 📂 Dataset Information  

**Dataset:** Enron Email Dataset  
- Consists of a large corpus of real emails from Enron employees  
- Includes both **spam** and **ham (non-spam)** labeled data  
- Contains email subjects and bodies in plain text  

---

## 🧠 Algorithms Used  

- **Naive Bayes Classifier:** A probabilistic model ideal for text classification  
- **Support Vector Machine (SVM):** A margin-based classifier suitable for high-dimensional data  

---

## 📊 Evaluation Metrics  

- **Accuracy:** Overall correctness of classification  
- **Precision:** Ability to avoid false positives (labeling ham as spam)  
- **Recall:** Ability to detect all spam emails (true positives)  

---

## 🏗️ Project Workflow  

1. **Data Collection:** Load and explore Enron dataset  
2. **Text Preprocessing:**  
   - Remove stop words, punctuation, and HTML  
   - Tokenization and lowercase conversion  
   - Vectorization using **TF-IDF** or **Count Vectorizer**  
3. **Model Training:** Train models using Naive Bayes and/or SVM  
4. **Model Evaluation:** Evaluate with test set using Accuracy, Precision, Recall  
5. **Optional Visualization:** Confusion matrix, word clouds for spam vs ham  

---

## 🔧 Setup Instructions  

### 1️⃣ Install Dependencies  
```bash
pip install numpy pandas scikit-learn nltk matplotlib seaborn
```

### 2️⃣ Download and Prepare the Dataset  
You can download the Enron dataset from sources like Kaggle or CMU. Organize emails into `spam/` and `ham/` folders.  

### 3️⃣ Run the Script  
```bash
python spam_detection.py
```

This will train and evaluate the model, showing classification performance.  

---

## 🚀 Future Enhancements  

- Experiment with **Deep Learning models** (LSTM, BERT) for improved accuracy  
- Deploy model via **Flask or Streamlit** as an interactive email filter  
- Integrate with a **real-time email service API** for live spam detection  

---

## 🙌 Credits  

- **Developer:** Shantanu
- **Dataset:** Enron Email Dataset  
- **Tech Stack:** Python, Scikit-learn, NLTK, Pandas  
