from flask import Flask,render_template,request,redirect
import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request
import pickle
from sklearn.preprocessing import LabelEncoder
import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "uVEty-CB4dYcccQ_Jq9V-atVXmL1dByE_wiDm95lcyTQ"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]
header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}


app = Flask(__name__)

@app.route('/',methods=["GET"])
def index():
    return render_template('index.html')


@app.route('/predict',methods=["POST","GET"])
def predict():
    if request.method == "POST":
        string = request.form['val']
        string = string.split(',')
        temp_input = [eval(i) for i in string]
        
        x_input = np.zeros(shape=(1, 10))
        x_input.shape
        
        lst_output = []
        n_steps = 10
        i=0
        while(i<10):
            if(len(temp_input)>10):
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1,-1)
                x_input = x_input.reshape((1,n_steps, 1))
                yhat = model.predict(x_input, verbose = 0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i=i+1
        
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose = 0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i=i+1
        
    
    	# NOTE: manually define and pass the array(s) of values to be scored in the next line
	payload_scoring = {"input_data": [{   "values": [[x_input]]    }]}

	response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/7f67cbed-6222-413b-9901-b2a72807ac82/predictions?version=2022-10-30', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
	predictions = response_scoring.json()
	print(response_scoring.json())
	
	val = lst_output[9]
	return render_template('web.html' , prediction = val)
    
    
    if request.method=="GET":
        return render_template('web.html')

if __name__=="__main__":
    model = load_model('C:/Users/rkara/IBM/Sprint - 4/Crude_oil.tar.gz')
    app.run(debug=True)