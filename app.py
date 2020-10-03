#import libraries

import flask
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template



app = flask.Flask(__name__, template_folder='templates')
model = None #global variable for the model

#function that loads the model
def load_model(model_):
    global model
    # model variable refers to the global variable
    model = joblib.load('iris_model.pkl', 'r') # no need to use with open . you can use the .pkl or sav extension with joblib
    return model  # returns the loaded model 


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
        arr_data = np.array(float_data).reshape(1,-1) #reshpae using (1,-1) for a single sample and (-1,1) for a single feature
        
        model_1 = load_model(model)  #assigns the returned model into the variable
        pred = model_1.predict(arr_data)
        
        return render_template('predict.html', prediction = pred[0])



if __name__ == "__main__":
    print("server started .....")
    
    load_model(model)

    print("model loaded .....")

    app.run(debug=True)
