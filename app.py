#import libraries

import flask
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template


model = None #global variable for the model

app = flask.Flask(__name__, template_folder='templates')

#function that loads the model
def load_model():
    global model
    # model variable refers to the global variable
    with open('iris_model.pkl', 'rb') as f:
        model = joblib.load(f)


#route to the homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    

#route to the predict page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
   
    if flask.request.method == 'POST':
        sepal_length = flask.request.form['sepal_length']
        sepal_width = flask.request.form['sepal_width']
        petal_length = flask.request.form['petal_length']
        petal_width = flask.request.form['petal_width']
        input_variables = [sepal_length, sepal_width, petal_length, petal_width]
        
        float_data = [float(x) for x in input_variables]
        arr_data = np.array(float_data).reshape(1, 4)
        
        pred = model.predict(arr_data[0])
        
        
    return render_template('predict.html', prediction = pred)



if __name__ == "__main__":
    print("server started .....")
    
    load_model()

    print("model loaded .....")

    app.run(debug=True)
