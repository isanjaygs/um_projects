import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('dataset.csv')

x = df.drop('price_range', axis = 1)
y = df['price_range']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(x_train_scaled,y_train)

y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test,y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

new_phone = [[1500, 1, 2.0, 1, 5, 1, 64, 0.5, 140, 4, 8, 1280, 1920, 4000, 12, 6, 15, 1, 1, 1]]

new_phone_scaled = scaler.transform(new_phone)

predicted_price_range = model.predict(new_phone_scaled)
print(predicted_price_range)