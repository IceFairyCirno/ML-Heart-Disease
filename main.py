import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("archive\Heart_disease_statlog.csv")

x = data.drop("target", axis=1)
y =  data["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)
print(f"This model's accuracy: {(accuracy_score(y_test, y_pred)*100):.2f}%")

info = input("Enter your age and sex (e.g. 60 M): ").split(" ")
age, sex = int(info[0]), (1 if info[1]==("M" or "m") else 0)

df=pd.DataFrame(x_train)
df = df[df['age'] == age]
df = df[df['sex'] == sex]

defaults = df.median()
defaults = pd.DataFrame(defaults).T

if defaults.isna().all().all() == True:
    print("No info related")
else:
    x_client = scaler.transform(defaults)
    prediction = model.predict(x_client)
    chance = model.predict_proba(x_client)
    print(f"Heart Disease: {"Yes" if prediction else "No"} with a chance of {(chance[:, (1 if prediction else 0)]*100)[0]:.1f}%")




