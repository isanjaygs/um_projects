import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('liver_cirrhosis.csv')

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:  # Find categorical columns
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert to numeric
    label_encoders[col] = le

x = df.drop('Stage',axis=1)
y = df['Stage']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=20)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = RandomForestClassifier(n_estimators=100,random_state=20)
model.fit(x_train_scaled,y_train)

y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test,y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

test_case = {
    "N_Days": 1500,
    "Status": "C",
    "Drug": "Placebo",
    "Age": 20000,
    "Sex": "M",
    "Ascites": "N",
    "Hepatomegaly": "Y",
    "Spiders": "N",
    "Edema": "N",
    "Bilirubin": 1.2,
    "Cholesterol": 250.0,
    "Albumin": 3.8,
    "Copper": 80.0,
    "Alk_Phos": 900.0,
    "SGOT": 65.0,
    "Tryglicerides": 70.0,
    "Platelets": 200.0,
    "Prothrombin": 10.5
}

test_df = pd.DataFrame([test_case])
for col in label_encoders:
    test_df[col] = label_encoders[col].transform(test_df[col])

test_df = test_df[x_train.columns]

test_df_scaled = scaler.transform(test_df)

predicted_stage = model.predict(test_df_scaled)
print(f"Predicted Stage: {predicted_stage[0]}")