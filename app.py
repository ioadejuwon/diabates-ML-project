from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler (you need to save and load the scaler along with the model)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Get the feature names from the scaler
feature_names = scaler.get_feature_names_out()
# print ("Feature Names: "+feature_names)

# Define the expected input features (excluding the target variable)
input_features = ['gender', 'age', 'hypertension', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/pred')
def predict_diabetes():
    return render_template('pred.html')

# @app.route('/result')
# def result():
#     return render_template('result.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = float(request.form['heart_disease'])
    smoking_history = int(request.form['smoking_history'])
    bmi = float(request.form['bmi'])
    
    HbA1c_level = float(request.form['HbA1c_level'])
    blood_glucose_level = int(request.form['blood_glucose_level'])


    # Standardize the input data
    input_data = np.array([[gender, age, hypertension, smoking_history, bmi, HbA1c_level, blood_glucose_level]])
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Map prediction outcome to human-readable labels
    if prediction == 0:
        prediction_label = 'No diabetes'
    # elif prediction == 1:
    #     prediction_label = 'Diabetes'
    else:
        prediction_label = 'Diabetes'

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
