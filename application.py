# flask , pandas ,scikit-learn , pickle-mixin
#flask-cors

from flask import Flask,render_template , request
import pandas as pd
import numpy as  np
import pickle

app = Flask(__name__)

model = pickle.load(open("LinearRegressionModel.pkl","rb"))
car = pd.read_csv("Cleaned car.csv")

@app.route("/")
def index():
    companies = sorted( car['company'].unique() )
    companies.insert(0,"Select Company")

    car_models = sorted( car['name'].unique() )
    year = sorted( car['year'].unique() , reverse=True )
    fuel_type = sorted( car['fuel_type'].unique() )

    return render_template('Index.html' , Companies = companies , Car_models = car_models , Years = year ,Fuel_type = fuel_type)

@app.route('/predict', methods=['post'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))

    # print(company,car_model,year,fuel_type,kms_driven)

    Prediction = model.predict(pd.DataFrame([[car_model , company , year , kms_driven ,fuel_type]], columns=['name', 'company' ,'year' ,'kms_driven','fuel_type']))

    # print(Prediction[0])

    return str(np.round(Prediction[0],2))

if __name__ == "__main__":
    app.run(debug=True)