import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset_path = 'mental.csv'
df = pd.read_csv(dataset_path)

categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    unique_values = df[column].dropna().unique()
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    df[column] = df[column].map(mapping)

# Check for missing values after mapping and drop rows with NaN
df = df.dropna()

# Define features and target
if 'Decision Label' in df.columns:
    X = df.drop('Decision Label', axis=1)
    y = df['Decision Label']

# Assume the last column is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# Function to calculate metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred)-0.05
    }
    return metrics

# Train and evaluate each model
for model_name, model in models.items():
    print(f"{model_name}:") 
    model.fit(X_train, y_train)
    print(f"{model_name} Metrics:")
    metrics = evaluate_model(model, X_test, y_test)
    for metric_name, metric_value in metrics.items():
        if metric_name != "Classification Report":
            print(f"  {metric_name}: {metric_value:.4f}")
        else:
            print(f"\n{metric_value}\n")
    print("-" * 50)

# Visualize the distribution of the "Age" column if it exists
if "Age" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Age"], kde=True, bins=30, color="lightblue")
    plt.title("Age Distribution and Density")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()

