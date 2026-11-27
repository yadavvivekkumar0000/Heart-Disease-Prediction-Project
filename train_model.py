import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Download the Cleveland Heart Disease dataset
print("Downloading dataset...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv(url, names=columns)

# 2. Handle missing values
df.replace("?", pd.NA, inplace=True)

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# 3. Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# 4. Convert target into binary classification
# 0 = No heart disease
# 1,2,3,4 = Heart disease
y = (y > 0).astype(int)

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)
print("Target classes:", y.unique())
print("Class distribution:")
print(y.value_counts())

# 5. Define categorical and numerical features
categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

numerical_features = [
    "age", "trestbps", "chol",
    "thalach", "oldpeak"
]

# 6. Preprocessing
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_features),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

# 7. Create model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

# 8. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Train model
print("\nTraining model...")
model.fit(X_train, y_train)

# 10. Evaluate model
predictions = model.predict(X_test)

print("\nModel Evaluation")
print("-------------------------")
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 11. Save model
joblib.dump(model, "heart_model.joblib")

print("\nModel saved successfully as heart_model.joblib")