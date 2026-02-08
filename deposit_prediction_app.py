import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
import pickle

# Create Flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict() # Extracts form data 
    X=pd.DataFrame([data])
    X=X.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(X)[0]
    return render_template(
    "index.html", prediction="Likely" if prediction=="yes" else "Not likely"
)

if __name__ == "__main__":
    app.run(debug=True)