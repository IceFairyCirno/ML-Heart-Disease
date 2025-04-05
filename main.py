import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=15)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model = LogisticRegression()
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
    return model, scaler, accuracy

def inquiry(defaults):
    questions = ["Any chest pain? (0: angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)",
                 "Resting blood pressure (mm Hg)",
                 "Cholesterol (mg/dl)",
                 "Fasting blood sugar > 120 mg/dl? (0: No, 1: Yes)",
                 "Resting ECG results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)",
                 "Maximum heart rate achieved",
                 "Exercise induced angina? (0: No, 1: Yes)",
                 "Oldpeak = ST depression induced by exercise relative to rest",
                 "Slope of the peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)",
                 "Number of major vessels colored by fluoroscopy (0-3)",
                 "Thalassemia (1: Normal, 2: Fixed defect, 3: Reversible defect)"]
    
    col = ["cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

    for i in range(len(questions)):
        print(questions[i])
        answer = input("Enter your answer (or press Enter to skip): ")
        if answer:
            try:
                defaults[col[i]] = float(answer)
            except ValueError:
                continue

    return defaults

def main():
    data = pd.read_csv("archive\Heart_disease_cleveland_new.csv")
    x = data.drop("target", axis=1)
    y =  data["target"]
    model, scaler, accuracy =train(x, y)
    print(f"This model's accuracy: {accuracy}%")

    user_info = input("Enter your age and sex (e.g. 60 M): ").split(" ")
    age, sex = int(user_info[0]), (1 if user_info[1]=="M" or "m" else 0)

    # Filtering the data based on user input
    df = pd.DataFrame(x)
    df = df[df['age'] == age]
    df = df[df['sex'] == sex]

    # Take the median of the filtered data
    defaults = df.median()
    defaults = pd.DataFrame(defaults).T

    # Adding user input for the rest of the features
    defaults = inquiry(defaults)

    if defaults.isna().all().all() == True:
        print("No info related")
    else:
        x_client = scaler.transform(defaults)
        prediction = model.predict(x_client)
        chance = model.predict_proba(x_client)
        print(f"There is a {(chance[:, (1 if prediction else 0)]*100)[0]:.1f}% chance that you are {"having" if prediction else "not having"} a heart disease")

main()











