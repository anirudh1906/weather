import pandas as pd
import pickle 
import numpy as np
from flask import Flask, redirect, url_for, render_template, request 

app= Flask(__name__)

model= pickle.load(open("C:\MLProjects\weather\model.pkl", 'rb'))

@app.route('/')
def welcome():
    return render_template('index.html') 


@app.route('/predict', methods=['POST'])
def predict():
    int_features= [int(x) for x in request.form.values()]
    prediction= model.predict(np.array(int_features).reshape(1,-1))
    
    return render_template("index.html", prediction_text= "the predicted weather is {}".format(prediction))

if __name__== '__main__':
    app.run(debug=True)