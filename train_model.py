import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================
# Load Dataset
# =====================================
data = pd.read_csv("startup_data.csv")

# Fill missing values
data = data.fillna(0)

# =====================================
# Target Variable (KEEP STRINGS)
# =====================================
y = data['status']   # 'acquired' and 'closed'

# =====================================
# Selected Features
# =====================================
features = [
    'age_first_funding_year',
    'age_last_funding_year',
    'age_first_milestone_year',
    'age_last_milestone_year',
    'relationships',
    'funding_rounds',
    'funding_total_usd',
    'milestones',
    'avg_participants'
]

X = data[features]

# =====================================
# Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=116
)

# =====================================
# Model
# =====================================
model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=116
)

model.fit(X_train, y_train)

# =====================================
# Evaluation
# =====================================
y_pred = model.predict(X_test)

print("====================================")
print("Model Evaluation")
print("====================================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
print("\nClasses:", model.classes_)

# =====================================
# Save Model
# =====================================
joblib.dump(model, "random_forest_model.pkl")

print("\nModel saved as random_forest_model.pkl")
