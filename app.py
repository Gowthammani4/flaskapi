
from flask import Flask,render_template,request,jsonify
#import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
@app.route('/api',methods=['POST'])
def predict():
    dataset = pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    yPrediction=regressor.predict([[float(request.args['exp'])]])
    print("post")

    print(yPrediction)
    return "Salary should be "+ str(yPrediction[0])

@app.route('/api',methods=['GET'])
def predictAPI():
    dataset = pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    yPrediction=regressor.predict([[float(request.args['exp'])]])
    print("Get")
    print(yPrediction)

    return str(yPrediction)

if __name__ == '__main__':
   app.run(debug=False,host='0.0.0.0')








