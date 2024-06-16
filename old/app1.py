# app.py
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define categorical feature names for one-hot encoding
gender_categories = ['Female', 'Male', 'Other']
smoking_history_categories = ['No Info', 'Current', 'Ever', 'Former', 'Never', 'Not Current']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data from the form
    age = float(request.form['age'])
    hypertension = float(request.form['hypertension'])
    heart_disease = float(request.form['heart_disease'])
    bmi = float(request.form['bmi'])
    HbA1c_level = float(request.form['HbA1c_level'])
    blood_glucose_level = float(request.form['blood_glucose_level'])
    gender = request.form['gender']
    smoking_history = request.form['smoking_history']

    # Convert categorical features to one-hot encoded format
    gender_encoded = [1 if gender == cat else 0 for cat in gender_categories]
    smoking_history_encoded = [1 if smoking_history == cat else 0 for cat in smoking_history_categories]

    # Concatenate all features into a numpy array
    input_data = np.array([age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level] + gender_encoded + smoking_history_encoded)

    # Reshape input data to match the shape expected by the model
    input_data = input_data.reshape(1, -1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction using the loaded model
    prediction = model.predict(input_data_scaled)

    # Render a template with the prediction
    return render_template('prediction.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
