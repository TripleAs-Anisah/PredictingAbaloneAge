from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.h5','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    length = float(request.form.get('length'))
    diameter = float(request.form.get('diameter'))
    height = float(request.form.get('height'))
    
    ##Test model prediction with static data. Reshape to change to 2D array 
    testdata = np.reshape([
    0,
    length,
    diameter,
    height,
    0,
    0,
    0,
    0
    ],(1, -1))

    pred_result = model.predict(testdata)

    return render_template('index.html', prediction_text='The predicted abalone age is: {:.2f} years.'.format(pred_result[0]))

if __name__ == "__main__":
    app.run()