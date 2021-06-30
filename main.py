import pickle
import sklearn
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

filename = 'static/model/SalPred_model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')
    if request.method == "POST":
        exp = request.form['experience']
        exp_ = np.array(exp)
        exp_ = exp_.reshape(-1,1)
        salary = model.predict(exp_)
        return render_template('index.html', pred=int(salary[0]), experience = exp)
        
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
