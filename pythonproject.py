import pandas as pd
import numpy as np

np.random.seed(42)

# Number of samples
n = 300

# Creating realistic petroleum engineering data
oil_rate = np.random.randint(500, 5000, n)        # STB/day
gas_rate = np.random.randint(100, 2000, n)        # Mscf/day
pressure = np.random.randint(1500, 4500, n)       # psi
temperature = np.random.randint(60, 180, n)       # °F
time = np.arange(1, n+1)

# Water cut formula with some noise (realistic behavior)
water_cut = (
    20
    + (pressure * 0.01)
    - (oil_rate * 0.002)
    + (gas_rate * 0.005)
    + np.random.normal(0, 5, n)
)

# Keep water cut in realistic range (0–100%)
water_cut = np.clip(water_cut, 0, 100)

# Create DataFrame
df = pd.DataFrame({
    "Oil_Rate": oil_rate,
    "Gas_Rate": gas_rate,
    "Pressure": pressure,
    "Temperature": temperature,
    "Time": time,
    "Water_Cut": water_cut
})

# Show first rows
print(df.head())

df.info()
df.head()

X = df[["Oil_Rate", "Gas_Rate", "Pressure", "Temperature", "Time"]]
y = df["Water_Cut"]

#train/testsplit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Evaluation
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("R2 Score:", r2)

#actual vs predicted
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(y_test.values, label="Actual Water Cut")
plt.plot(y_pred, label="Predicted Water Cut")
plt.legend()
plt.title("Actual vs Predicted Water Cut")
plt.xlabel("Samples")
plt.ylabel("Water Cut (%)")
plt.show(block=False)
plt.pause(2)
plt.savefig("graph1.png")
plt.close()


import numpy as np

errors = y_test - y_pred

plt.figure(figsize=(8,5))
plt.scatter(range(len(errors)), errors)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residual Errors")
plt.xlabel("Sample Index")
plt.ylabel("Error")
plt.show(block=False)
plt.pause(2)
plt.savefig("graph2.png")
plt.close()


plt.savefig("actual_vs_predicted_watercut.png")



print("Starting Random Forest...")

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)


from sklearn.metrics import r2_score, mean_absolute_error

print("Linear Regression R2:", r2_score(y_test, y_pred))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))

print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred))
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))


#input prediction
print("\n--- Water Cut Predictor ---")
oil = float(input("Enter Oil Rate (STB/day): "))
gas = float(input("Enter Gas Rate (Mscf/day): "))
pres = float(input("Enter Pressure (psi): "))
temp = float(input("Enter Temperature (°F): "))
time = float(input("Enter Time (days): "))

user_input = [[oil, gas, pres, temp, time]]
prediction = model.predict(user_input)
print(f"Predicted Water Cut: {prediction[0]:.2f}%")


# Pie chart for prediction
categories = ['Water %', 'Oil %']
values = [prediction[0], 100 - prediction[0]]
colors = ['blue', 'green']

plt.figure(figsize=(6,6))
plt.pie(values, labels=categories, colors=colors,
        autopct='%1.1f%%', startangle=90)
plt.title(f'Well Production: Water Cut = {prediction[0]:.2f}%')
plt.show()



