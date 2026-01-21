"""
Titanic Survival Prediction - Model Development
================================================
This script loads the Titanic dataset, performs preprocessing,
trains a Random Forest Classifier, and saves the model for deployment.

Selected Features (5): Pclass, Sex, Age, SibSp, Fare
Target Variable: Survived
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ============================================
# 1. Load the Dataset
# ============================================
print("=" * 50)
print("TITANIC SURVIVAL PREDICTION - MODEL DEVELOPMENT")
print("=" * 50)

# Load the training dataset
df = pd.read_csv('train.csv')

print("\n📊 Dataset Overview:")
print(f"   Total records: {len(df)}")
print(f"   Total features: {len(df.columns)}")
print(f"\n   Columns: {list(df.columns)}")

# ============================================
# 2. Data Preprocessing
# ============================================
print("\n" + "=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)

# 2a. Handle Missing Values
print("\n🔍 Missing Values Before Handling:")
print(df[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']].isnull().sum())

# Fill missing Age values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Fare values with median (if any)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

print("\n✅ Missing Values After Handling:")
print(df[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']].isnull().sum())

# 2b. Feature Selection (5 features as required)
selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
target = 'Survived'

print(f"\n📋 Selected Features: {selected_features}")
print(f"📋 Target Variable: {target}")

# Create feature matrix and target vector
X = df[selected_features].copy()
y = df[target].copy()

# 2c. Encode Categorical Variables
print("\n🔄 Encoding Categorical Variables...")
label_encoder = LabelEncoder()
X['Sex'] = label_encoder.fit_transform(X['Sex'])  # male=1, female=0
print(f"   Sex encoding: female=0, male=1")

# 2d. Feature Scaling
print("\n📏 Applying Feature Scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"   Scaled features shape: {X_scaled.shape}")

# ============================================
# 3. Train-Test Split
# ============================================
print("\n" + "=" * 50)
print("TRAIN-TEST SPLIT")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\n   Training set size: {len(X_train)}")
print(f"   Testing set size: {len(X_test)}")

# ============================================
# 4. Train Random Forest Classifier
# ============================================
print("\n" + "=" * 50)
print("MODEL TRAINING - RANDOM FOREST CLASSIFIER")
print("=" * 50)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("\n🚀 Training the model...")
model.fit(X_train, y_train)
print("✅ Model training completed!")

# ============================================
# 5. Model Evaluation
# ============================================
print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📈 Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification Report
print("\n📊 Classification Report:")
print("-" * 50)
print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))

# Feature Importance
print("\n🔑 Feature Importance:")
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in feature_importance.iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.4f}")

# ============================================
# 6. Save the Model (Using Joblib)
# ============================================
print("\n" + "=" * 50)
print("SAVING MODEL")
print("=" * 50)

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the model
model_path = 'model/titanic_survival_model.pkl'
joblib.dump(model, model_path)
print(f"\n✅ Model saved to: {model_path}")

# Save the scaler (needed for preprocessing new data)
scaler_path = 'model/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved to: {scaler_path}")

# Save the label encoder (needed for Sex encoding)
encoder_path = 'model/label_encoder.pkl'
joblib.dump(label_encoder, encoder_path)
print(f"✅ Label Encoder saved to: {encoder_path}")

# ============================================
# 7. Demonstrate Model Reload and Prediction
# ============================================
print("\n" + "=" * 50)
print("MODEL RELOAD DEMONSTRATION")
print("=" * 50)

# Reload the saved model
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)
loaded_encoder = joblib.load(encoder_path)
print("\n✅ Model, Scaler, and Encoder reloaded successfully!")

# Test prediction with sample data
print("\n🧪 Testing with sample passenger data:")
sample_passenger = {
    'Pclass': 1,
    'Sex': 'female',
    'Age': 25,
    'SibSp': 1,
    'Fare': 100.0
}

print(f"   Input: {sample_passenger}")

# Preprocess the sample data
sample_df = pd.DataFrame([sample_passenger])
sample_df['Sex'] = loaded_encoder.transform(sample_df['Sex'])
sample_scaled = loaded_scaler.transform(sample_df)

# Make prediction
prediction = loaded_model.predict(sample_scaled)
prediction_proba = loaded_model.predict_proba(sample_scaled)

result = "Survived ✅" if prediction[0] == 1 else "Did Not Survive ❌"
confidence = max(prediction_proba[0]) * 100

print(f"\n   Prediction: {result}")
print(f"   Confidence: {confidence:.2f}%")

print("\n" + "=" * 50)
print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
print("=" * 50)
