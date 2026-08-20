from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json

app = Flask(__name__)

# Load the trained model
model = joblib.load('logistic_regression_model.joblib')

# Load the feature columns used during training
with open('feature_columns.json', 'r') as f:
    model_feature_columns = json.load(f)

current_categorical_cols = ['Region', 'Country', 'City', 'Claim_Type', 'Driver_Gender', 'Driving_History', 'Vehicle_Make', 'Vehicle_Model', 'Vehicle_Usage', 'Policy_Coverage_Type', 'Accident_Prone_Area', 'Police_Report_Filed', 'Fraud_Suspected', 'Third_Party_Involved', 'Claim_Channel', 'Adjuster_ID', 'Road_Type', 'Weather_Condition']
current_numerical_cols = ['Claim_Amount_Requested_USD', 'Claim_Amount_Approved_USD', 'Driver_Age', 'Driver_License_Years', 'Previous_Claims_Count', 'Vehicle_Year', 'Vehicle_Value_USD', 'Annual_Premium_USD', 'Deductible_USD', 'Claim_Processing_Days']
required_cols = current_categorical_cols + current_numerical_cols


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    missing = [c for c in required_cols if c not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    input_df = pd.DataFrame([data])

    input_encoded = pd.get_dummies(input_df[current_categorical_cols], columns=current_categorical_cols, drop_first=True)
    processed_input = pd.concat([input_encoded, input_df[current_numerical_cols]], axis=1)
    processed_input = processed_input.reindex(columns=model_feature_columns, fill_value=0)

    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    return jsonify({
        "prediction": int(prediction[0]),
        "probability_class_0": float(prediction_proba[0][0]),
        "probability_class_1": float(prediction_proba[0][1])
    })


if __name__ == '__main__':
    # Local/server run. For Colab, replace this block with flask_ngrok as before.
    app.run(debug=False, host='0.0.0.0', port=5000)
