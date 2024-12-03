from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
with open('churn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the form
        data = request.get_json()
        features = [
            float(data['Age']),
            float(data['Tenure']),
            float(data['MonthlyCharges']),
            float(data['ContractType']),
            float(data['InternetService']),
            float(data['TotalCharges']),
            float(data['TechSupport'])
        ]
        
        # Scale the input data
        scaled_features = scaler.transform([features])
        
        # Predict churn
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)[0][1]  # Probability of churn

        # Map prediction to output
        output = "Yes" if prediction[0] == 1 else "No"

        return jsonify({
            "Churn Prediction": output,
            "Probability": f"{probability:.2%}"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
