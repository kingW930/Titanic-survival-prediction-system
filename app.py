"""
Titanic Survival Prediction - Flask Web Application
====================================================
A web-based GUI for predicting Titanic passenger survival.
"""

from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessors
model = joblib.load('model/titanic_survival_model.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

@app.route('/')
def home():
    """Render the home page with the prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process the form data and return the prediction."""
    try:
        # Get form data
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        fare = float(request.form['fare'])
        
        # Create DataFrame with the input
        input_data = pd.DataFrame([{
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Fare': fare
        }])
        
        # Encode Sex
        input_data['Sex'] = label_encoder.transform(input_data['Sex'])
        
        # Scale the features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        # Prepare result
        survived = prediction[0] == 1
        confidence = max(prediction_proba[0]) * 100
        
        result = {
            'survived': survived,
            'message': 'Survived' if survived else 'Did Not Survive',
            'confidence': f'{confidence:.1f}%',
            'input': {
                'pclass': pclass,
                'sex': sex.capitalize(),
                'age': age,
                'sibsp': sibsp,
                'fare': fare
            }
        }
        
        return render_template('index.html', result=result)
        
    except Exception as e:
        error = {'message': f'Error: {str(e)}'}
        return render_template('index.html', error=error)

if __name__ == '__main__':
    print("=" * 50)
    print("TITANIC SURVIVAL PREDICTION SYSTEM")
    print("=" * 50)
    print("\n🚀 Starting Flask server...")
    print("📍 Open http://localhost:5000 in your browser")
    print("\nPress Ctrl+C to stop the server\n")
    app.run(debug=True, port=5000)
