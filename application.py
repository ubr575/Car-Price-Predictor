from urllib import request

from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

model=pickle.load(open('D:/python projects/Car Price Predictor/LinearRegModel.pkl','rb'))
car=pd.read_csv('D:/python projects/Car Price Predictor/Cleaned car.csv')
app = Flask(__name__)

@app.route('/')
def index():
    company_name=sorted(car['company'].unique())
    company_model=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    kms_driven=sorted(car['kms_driven'].unique())
    fuel_type=car['fuel_type'].unique()
    company_name.insert(0,"Select Company")
    return render_template('index.html',companies=company_name,company_names=company_model,years=year,kms_driven=kms_driven,fuel_type=fuel_type)

@app.route('/predict',methods=['POST'])
def predict():
    company=request.form.get('company')
    car_model=request.form.get('car_model')
    year=int(request.form.get('year'))
    kms_driven=int(request.form.get('kilo_driven'))
    fuel_type = request.form.get('fuel_type')
    # print(company,car_model,year,kms_driven,fuel_type)
    prediction=model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
    # print(prediction)
    return str(np.round(prediction[0],2))
if __name__ == '__main__':
    app.run(debug=True)