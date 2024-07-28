from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('fish_weight_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    length1 = float(request.form['length1'])
    length2 = float(request.form['length2'])
    length3 = float(request.form['length3'])
    height = float(request.form['height'])
    width = float(request.form['width'])

    features = np.array([[length1, length2, length3, height, width]])
    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    return render_template('prediction.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
