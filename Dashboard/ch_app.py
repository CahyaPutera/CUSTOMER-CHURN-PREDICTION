from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import json

# Initiate Flask
app = Flask(__name__)

# Model
model = joblib.load('LRG_model')

# Root
@app.route('/')
def route_root():
    return render_template('home.html')

# Homepage
@app.route('/home')
def route_home():
    return render_template('home.html')

# Dataset
@app.route('/dataset')
def route_dataset():
    return render_template('dataset.html')

# Prediction
@app.route('/predict', methods = ['GET'])
def route_predict():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def route_result():
    if request.method == 'POST':

        contract = int(request.form['contract'])
        security = int(request.form['security'])
        support = int(request.form['support'])
        fiber = int(request.form['fiber'])
        check = int(request.form['check'])
        backup = int(request.form['backup'])
        protection = int(request.form['protection'])
        paperless = int(request.form['paperless'])
        contract2year = int(request.form['contract2year'])
        internet = int(request.form['internet'])

        prob = round(model.predict_proba([[contract,security,support,fiber,check,backup,protection,paperless,contract2year,internet]])[0][1]*100)
        result = 'The probability of Churn is : {}%'.format(prob)
    
    return render_template('predict.html', results = result)

if __name__ == "__main__":
    app.run(debug = True)
