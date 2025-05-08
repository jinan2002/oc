from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import gzip  # ضروري لأن الملف مضغوط

app = Flask(__name__)

# تحميل النموذج من ملف .pkl.gz المضغوط
with gzip.open('random_forest_model.pkl.gz', 'rb') as f:
    model = joblib.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    city = request.form['city']
    region = request.form['region']
    property_type = request.form['property_type']

    # تحويل البيانات إلى DataFrame مثل وقت التدريب
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
