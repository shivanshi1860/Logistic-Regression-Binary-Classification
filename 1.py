
# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()  # ✅ Correct syntax

# Convert dataset to DataFrame for better understanding
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

print("✅ Dataset Loaded Successfully!")
print("Shape of X:", X.shape)
print("Unique target values:", np.unique(y))
print("\nFirst 5 rows:\n", X.head())

# Step 3: Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n✅ Data Split Done!")
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Step 4: Initialize and Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\n✅ Model Training Complete!")

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate Model
print("\n===== MODEL EVALUATION =====")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix")
plt.show()

# Step 8 (Optional): Save the Model
import joblib
joblib.dump(model, "logistic_regression_model.pkl")
print("\n✅ Model Saved Successfully as 'logistic_regression_model.pkl'")
