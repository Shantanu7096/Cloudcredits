import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load Dataset with File Check
file_path = r"D:\GitHub\cloudcredits\Project 7\email.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    print("Loaded dataset with shape:", df.shape)
else:
    print(f"Error: File '{file_path}' not found.")
    exit()

# Rename & Clean Data
df.rename(columns={'Category': 'label', 'Message': 'text'}, inplace=True)

# Convert 'ham' to 0 and 'spam' to 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop missing values
df.dropna(subset=['text', 'label'], inplace=True)

# Verify Data Integrity
if df.empty:
    print("Error: No valid data after cleaning.")
    exit()

# Split Data
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)

# Train SVM Model
svm_model = LinearSVC(dual=False)
svm_model.fit(X_train_vec, y_train)
svm_preds = svm_model.predict(X_test_vec)

# Model Evaluation Function
def evaluate_model(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=1))
    print("Recall   :", recall_score(y_true, y_pred, zero_division=1))

evaluate_model("Naive Bayes", y_test, nb_preds)
evaluate_model("SVM", y_test, svm_preds)
