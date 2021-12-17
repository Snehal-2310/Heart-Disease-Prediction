# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('chest pain type')
        trestbps = int(request.form['resting bp s'])
        chol = int(request.form['cholesterol'])
        fbs = request.form.get('fasting blood sugar')
        restecg = int(request.form['resting ecg'])
        thalach = int(request.form['max heart rate'])
        exang = request.form.get('exercise angina')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('ST slope')
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(debug=True)

