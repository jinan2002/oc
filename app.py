from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import gdown
import os

model_path = "random_forest_model.pkl"

if not os.path.exists(model_path):
    url = 'https://drive.google.com/uc?id= https://drive.google.com/file/d/1f9ZTqiMxaDCaBFZ85BS41hMpYyif0QBe '  # استبدل هذا بالـ ID الحقيقي
    gdown.download(url, model_path, quiet=False)


app = Flask(__name__)
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    city = request.form['city']
    region = request.form['region']
    property_type = request.form['property_type']

    # حول البيانات إلى DataFrame كما استخدمته أثناء التدريب
    input_df = pd.DataFrame([{
        'area': area,
        'city': city,
        'region': region,
        'property_type': property_type
    }])

    prediction = model.predict(input_df)[0]
    return f"<h2>💰 Predicted Price: {prediction:.2f}</h2>"

if __name__ == '__main__':
    app.run(debug=True)
