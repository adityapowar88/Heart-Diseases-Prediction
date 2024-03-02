#from flask import Flask, render_template, request
#import pickle
#import numpy as np

#model = pickle.load(open('hateSpeechModel.pickle', 'rb'))

#app = Flask(__name__)


#@app.route('/')
#def man():
   # return render_template('home.html')

#@app.route('/predict', methods=['POST'])
#def home():
 #text1 = request.form['tweet']

 #pred = model.predict(text1)
 #return render_template('after.html', data=pred)


#if __name__ =="__main__":
 #   app.run(debug=True)


from flask import Flask, render_template,request, jsonify
import numpy as np
import pickle
import re
import preprocessor as p
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)

import pandas as pd



@app.route("/")
def hello():
    return render_template("home.html")


def rfc_model(data):
    filename_model = 'final_predict_model.sav'
    model = clf2 = pickle.load(open(filename_model,'rb'))
    pred=model.predict(data)[0]
    if pred==1:
        return 'Stroke Occure'
    else:
        return 'Stroke not occure'

@app.route("/submit", methods =['POST'])
def submit():
    if request.method == "POST":  
        age = request.form['age']
        hypertension = request.form['hypertension']
        ever_married = request.form['ever_married']
        gender = request.form['gender']
        work_type = request.form['work_type']
        Residence_type = 1
        avg_glucoose_level = request.form['avg_glucoose_level']
        bmi = request.form['bmi']
        result = rfc_model([[gender,age,hypertension,1,ever_married,work_type,Residence_type,avg_glucoose_level,bmi]])
        
        return render_template("submit.html",result=result)


if __name__ == "__main__":
    app.run(debug=True) 
