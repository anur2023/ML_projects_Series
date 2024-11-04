from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
with open('../model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        blood_pressure = request.form['blood_pressure']
        skin_thickness = request.form['skin_thickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        diabetes_pedigree_function = request.form['diabetes_pedigree_function']
        age = request.form['age']
        features = [[
            float(pregnancies), 
            float(glucose), 
            float(blood_pressure), 
            float(skin_thickness), 
            float(insulin), 
            float(bmi), 
            float(diabetes_pedigree_function), 
            float(age)
        ]]
        prediction = model.predict(features)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
