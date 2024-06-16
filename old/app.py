from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
with open('diabetes_2_LR2.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler (you need to save and load the scaler along with the model)
with open('min_max_scaler2.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the expected input features (excluding the target variable)
input_features = ['Age', 'Sex', 'HighChol', 'BMI', 'PhysActivity', 'PhysHlth']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    high_chol = int(request.form['high_chol'])
    bmi = float(request.form['bmi'])
    phys_activity = int(request.form['phys_activity'])
    phys_hlth = float(request.form['phys_hlth'])

    # Standardize the input data
    input_data = np.array([[age, sex, high_chol, bmi, phys_activity, phys_hlth]])
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Map prediction outcome to human-readable labels
    if prediction == 0:
        prediction_label = "No diabetes"
    else:
        prediction_label = "Diabetes"

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
