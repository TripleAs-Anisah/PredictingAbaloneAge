from flask import Flask, request, jsonify, url_for, redirect, render_template
import numpy as np
import pickle
import requests

app = Flask(__name__)
model = pickle.load(open('model.h5','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
   
    length = request.form.get('Length')
    diameter = request.form.get('Diameter')
    height = request.form.get('Height')
    # LoanDuration = request.form.get('duration')              
   
    ##Test model prediction with static data. Reshape to change to 2D array 
    testdata = np.reshape([
    None,
    length,
    diameter,
    height,
    None,
    None,
    None,
    None
    ],(1, -1))

    pred_result = model.predict(testdata)

    
    return render_template('index.html', prediction_text='The predicted abalone age is: {}.2f.'.format(pred_result))

if __name__ == "__main__":
    app.run()
