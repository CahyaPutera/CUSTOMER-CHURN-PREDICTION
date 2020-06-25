# Importing Libraries 
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

# Import Dataset
df = pd.read_csv('df_ready.csv')

# Split data 90% Train
x = df[['Contract_Month-to-month', 'OnlineSecurity_No', 'TechSupport_No', 'InternetService_Fiber optic', 'PaymentMethod_Electronic check', 'OnlineBackup_No', 'DeviceProtection_No', 'PaperlessBilling_Yes', 'Contract_Two year', 'Tenure', 'MonthlyCharges']]
y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle = False)

# SMOTE
x_train_sm, y_train_sm = SMOTE(random_state = False).fit_sample(x_train, y_train.ravel())

# Applying model
model =  joblib.load('modelKNN')

# Applying scaler
scaler = MinMaxScaler().fit_transform(df[['Tenure', 'MonthlyCharges']])

# Initiate Flask
app = Flask(__name__)

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
        contract2y = int(request.form['contract2y'])
        tenure = int(request.form['tenure'])
        charges = float(request.form['charges'])

        prob = round(model.predict_proba([[contract,security,support,fiber,check,backup,protection,paperless,contract2y,tenure,charges]])[0][1]*100, 2)

        if prob < 50 :
            result = 'The Probability of Churn is {}% - This Customers Might Not Churn'.format(prob)
            
        else:
            result = 'The Probability of Churn is {}% - This Customers Might Churn'.format(prob)
        
        return render_template('predict.html', results = result)

if __name__ == "__main__":
    app.run(debug = True)