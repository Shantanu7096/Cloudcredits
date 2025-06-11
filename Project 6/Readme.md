# 🎭 Sentiment Analysis on Movie Reviews  

## 📌 Project Overview  

This project aims to **classify movie reviews as either positive or negative** using machine learning techniques. By analyzing the textual content of user reviews from the **IMDb Movie Reviews Dataset**, we train models to detect sentiment patterns using both traditional and deep learning approaches.  

---

## 📂 Dataset Information  

**Dataset:** IMDb Movie Reviews Dataset  
**Data Size:** 50,000 movie reviews (25,000 labeled for training, 25,000 for testing)  
**Labels:**  
- **1:** Positive Review  
- **0:** Negative Review  

---

## 🧠 Algorithms  

- **Naive Bayes:** A fast, probabilistic model often used for text classification.  
- **LSTM (Long Short-Term Memory):** A deep learning model that captures sequential dependencies in text data—great for context-aware sentiment analysis.  

---

## 📊 Evaluation Metrics  

- **Accuracy:** Measures the overall percentage of correctly classified reviews.  
- **F1 Score:** Harmonic mean of precision and recall; balances false positives and false negatives.  

---

## 🛠️ Project Workflow  

1. **Load and Clean Data:** Remove HTML tags, stopwords, and punctuation.  
2. **Tokenization & Vectorization:** Convert text into numerical format using techniques like TF-IDF (for Naive Bayes) or Embeddings (for LSTM).  
3. **Train-Test Split:** Split the dataset into training and testing sets.  
4. **Model Training:** Train models using Naive Bayes and LSTM.  
5. **Model Evaluation:** Compute accuracy and F1 score.  
6. **Visualizations (Optional):** Show word frequency, accuracy comparison, or confusion matrix.  

---

## 🔧 Setup Instructions  

### 1️⃣ Install Dependencies  
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras nltk
```

### 2️⃣ Run the Script  
```bash
python sentiment_analysis.py
```
##Screenshot - Output
```
![image](https://github.com/user-attachments/assets/c20c92d5-fe94-4bb5-a9ea-51c838580c73)

```
---

## 🚀 Future Enhancements  

- Use **BERT** or transformer-based models for state-of-the-art accuracy.  
- Add **Streamlit or Flask** to deploy the sentiment classifier as a web app.  
- Incorporate **multilingual sentiment detection** for global reviews.  

---

## 🙌 Credits  

- **Developer:** Shantanu 
- **Dataset Source:** IMDb Movie Reviews Dataset  
- **Tools & Libraries:** Python, Scikit-learn, TensorFlow, Keras, NLTK  
