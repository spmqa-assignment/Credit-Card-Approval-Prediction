import pandas
from matplotlib import pyplot as plt
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
inp = scaler.transform([[0,85,22,0,0,7,0,0,1,0,2,0,0,0]])
svc.predict(inp)[0]