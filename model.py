# ==============================================
# PART A: Mental Health Numerical Data Model
# ==============================================

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
mental_df = pd.read_csv("Datasets/mental.csv")

# Handle missing values
mental_df.fillna(method='ffill', inplace=True)

# Encode categorical variables
label_encoders = {}
for col in mental_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    mental_df[col] = le.fit_transform(mental_df[col])
    label_encoders[col] = le

# Define features and target
X = mental_df.drop(columns=['Decision Label'])
y = mental_df['Decision Label']

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
mental_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
mental_rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = mental_rf_model.predict(X_test)
print("\n--- Mental Health Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model and encoders
with open("mental_health_rf_model.pkl", "wb") as file:
    pickle.dump(mental_rf_model, file)

with open("label_encoders.pkl", "wb") as le_file:
    pickle.dump(label_encoders, le_file)

print("Mental health model and encoders saved successfully!")


# ==============================================
# PART B: HR-Based Text Data Model
# ==============================================

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load HR text dataset
hr_df = pd.read_csv("Datasets/Combined Data.csv")

# Preprocess text
hr_df['statement'] = hr_df['statement'].fillna('')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

hr_df['cleaned_statement'] = hr_df['statement'].apply(preprocess_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

hr_df['cleaned_statement'] = hr_df['cleaned_statement'].apply(remove_stopwords)

# Handle empty cleaned statements
hr_df['cleaned_statement'] = hr_df['cleaned_statement'].fillna('')

# Define features and target
X_text = hr_df['cleaned_statement']
y_text = hr_df['status']

# Convert labels to integers
label_map = {label: idx for idx, label in enumerate(y_text.unique())}
y_text_int = y_text.map(label_map)

# Train-test split
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text_int, test_size=0.2, random_state=42)

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# Train Random Forest on text
hr_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
hr_rf_model.fit(X_train_tfidf, y_train_text)

# Evaluate text model
y_pred_text = hr_rf_model.predict(X_test_tfidf)
print("\n--- HR Text Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test_text, y_pred_text))
print("\nClassification Report:\n", classification_report(y_test_text, y_pred_text))
print("Confusion Matrix:\n", confusion_matrix(y_test_text, y_pred_text))

# Save the model and vectorizer
with open("text_rf_model.pkl", "wb") as model_file:
    pickle.dump(hr_rf_model, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("HR Text model and vectorizer saved successfully!")
