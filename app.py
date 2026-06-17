import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Water Cut Predictor", page_icon="🛢️")

st.title("🛢️ Water Cut Prediction App")
st.write("Petroleum Engineering Machine Learning Project")

# Generate training data
np.random.seed(42)
n = 300
oil_rate = np.random.randint(500, 5000, n)
gas_rate = np.random.randint(100, 2000, n)
pressure = np.random.randint(1500, 4500, n)
temperature = np.random.randint(60, 180, n)
time = np.arange(1, n+1)

water_cut = (
    20
    + (pressure * 0.01)
    - (oil_rate * 0.002)
    + (gas_rate * 0.005)
    + np.random.normal(0, 5, n)
)
water_cut = np.clip(water_cut, 0, 100)

df = pd.DataFrame({
    "Oil_Rate": oil_rate,
    "Gas_Rate": gas_rate,
    "Pressure": pressure,
    "Temperature": temperature,
    "Time": time,
    "Water_Cut": water_cut
})

X = df[["Oil_Rate", "Gas_Rate", "Pressure", "Temperature", "Time"]]
y = df["Water_Cut"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ---------------------------------------
# SECTION 1: MODEL PERFORMANCE (shown once, at the top)
# ---------------------------------------
st.header("📊 Model Performance")

col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2:.2f}")
col2.metric("MAE", f"{mae:.2f}%")

st.subheader("Actual vs Predicted Water Cut")
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(y_test.values, label="Actual Water Cut", color="#3498db")
ax1.plot(y_pred, label="Predicted Water Cut", color="#e67e22")
ax1.set_xlabel("Samples")
ax1.set_ylabel("Water Cut (%)")
ax1.legend()
st.pyplot(fig1)

st.subheader("Residual Errors")
errors = y_test.values - y_pred
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.scatter(range(len(errors)), errors, color="#3498db", alpha=0.6)
ax2.axhline(y=0, color="red", linestyle="--")
ax2.set_xlabel("Sample Index")
ax2.set_ylabel("Error")
st.pyplot(fig2)

st.write("---")

# ---------------------------------------
# SECTION 2: TRY IT YOURSELF (interactive prediction)
# ---------------------------------------
st.header("🎯 Try It Yourself")

st.sidebar.header("Enter Well Parameters")
oil = st.sidebar.number_input("Oil Rate (STB/day)", 500, 5000, 2000)
gas = st.sidebar.number_input("Gas Rate (Mscf/day)", 100, 2000, 500)
pres = st.sidebar.number_input("Pressure (psi)", 1500, 4500, 3000)
temp = st.sidebar.number_input("Temperature (°F)", 60, 180, 120)
t = st.sidebar.number_input("Time (days)", 1, 300, 50)

if st.sidebar.button("Predict Water Cut"):
    user_input = [[oil, gas, pres, temp, t]]
    prediction = model.predict(user_input)[0]

    st.success(f"### Predicted Water Cut: {prediction:.2f}%")

    fig3, ax3 = plt.subplots()
    values = [prediction, 100 - prediction]
    labels = ['Water %', 'Oil %']
    colors = ['#3498db', '#2ecc71']
    ax3.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title("Well Production Breakdown")
    st.pyplot(fig3)
else:
    st.info("Enter values in the sidebar and click 'Predict Water Cut' to see results.")

st.write("---")
st.write("Built with Python, scikit-learn & Streamlit")