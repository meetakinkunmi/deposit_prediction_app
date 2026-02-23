import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
import matplotlib.pyplot as plt
import io
import base64
import joblib

# Create Flask app
app = Flask(__name__)

# Load the pickle model
model = joblib.load(open('model.pkl.z', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict() # Extracts data from the form 
    X=pd.DataFrame([data])  # Convert input data into DataFrame as required
    X=X.reindex(columns=model.feature_names_in_, fill_value=0) # Make sure variable names align with the model's
    prediction = model.predict_proba(X)[0][1]
    predidction_message = "Likely" if prediction > 0.5 else "Not likely"
    prediction_confidence = round(prediction * 100, 2)

    # Getting the Features Importance chart
    importances = model.named_steps['classifier'].feature_importances_
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    featured_var = model.named_steps['selector'].get_support()
    final_features = feature_names[featured_var]

    plt.figure(figsize=(8,4))
    pd.Series(importances, index=final_features).nlargest(10).plot(
        kind='barh', color='skyblue', edgecolor='navy')
    plt.title("Top 10 Infuential Features")

    # Saving the chart to a buffer
    image = io.BytesIO()
    plt.savefig(image, format='png', bbox_inches = 'tight')
    image.seek(0)
    chart_url = base64.b64encode(image.getvalue()).decode()
    plt.close()

    return render_template(
    "index.html", prediction=predidction_message,
    probability=prediction_confidence,
    plot_url = f"data:image/png;base64,{chart_url}"
    )

if __name__ == "__main__":
    app.run(debug=True)