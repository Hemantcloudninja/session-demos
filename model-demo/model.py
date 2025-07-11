import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create the dataset
data = {
    "Income": [50000, 60000, 25000, 40000, 80000],
    "CreditScore": [700, 650, 600, 620, 750],
    "LoanAmount": [200000, 250000, 100000, 150000, 300000],
    "LoanTerm": [360, 360, 180, 240, 360],
    "Approved": ["Yes", "Yes", "No", "No", "Yes"]
}

# Step 2: Convert data to DataFrame
df = pd.DataFrame(data)

# Step 3: Encode the target column
le = LabelEncoder()
df['Approved'] = le.fit_transform(df['Approved'])  # Yes=1, No=0

# Step 4: Split features and target
X = df[["Income", "CreditScore", "LoanAmount", "LoanTerm"]]
y = df["Approved"]

# Step 5: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 6: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 8: Save the model to a file
model_path = "./model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
