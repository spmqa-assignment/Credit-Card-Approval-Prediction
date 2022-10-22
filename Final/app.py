
from flask import Flask, render_template, request


import pandas
import numpy as np
df = pandas.read_csv("clean_dataset1.csv")

from sklearn.preprocessing import LabelEncoder,StandardScaler

encoder = LabelEncoder().fit(df['Industry'])

df['Industry'] = encoder.transform(df['Industry'])

encoder = LabelEncoder().fit(df['Citizen'])


df['Citizen'] = encoder.transform(df['Citizen'])


encoder = LabelEncoder().fit(df['Ethnicity'])


df['Ethnicity'] = encoder.transform(df['Ethnicity'])

df["PriorDefault"] = 1-df["PriorDefault"]


X = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.svm import SVC
from sklearn.metrics import f1_score

svc = SVC()

svc.fit(X_train_scaled, y_train)

y_pred_svc = svc.predict(X_test_scaled)

print(f"f1 score using svc = {np.round(f1_score(y_test, y_pred_svc),2)}")

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/result", methods = ['POST','GET'])
def result():
    output = request.form.to_dict()
    print(output)
    name = output["name"]
    age = output["age"]
    Ethnicity = output["Ethnicity"]
    gender = output["gender"]
    m_status = output["m_status"]
    citizen = output["citizen"]
    defauter = output["defauter"]
    Industry = output["Industry"]
    employed = output["employed"]
    customer = output["customer"]
    debt = output["debt"]
    credits = output["credits"]
    income = output["income"]
    years = output["years"]
    driver_license = output["driver_license"]
    inp = scaler.transform([[gender,age,debt,m_status,customer,Industry,Ethnicity,years,defauter,employed,credits,driver_license,citizen,income]])
    pre = svc.predict(inp)[0]
    if pre == 0:
        msg = "Sorry Your Card Won't be Approved"
    elif pre == 1:
        msg = "Congrats Your Card Will be Approved"
    return render_template("result.html", name = name, msg=msg)

if __name__ == '__main__':
    app.run(debug=True, port=5001)