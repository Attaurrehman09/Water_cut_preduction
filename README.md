# 🛢️ Water Cut Prediction System

A machine learning project that predicts water cut percentage in oil wells using petroleum engineering parameters like oil rate, gas rate, pressure, and temperature.

## 📌 About This Project

I built this project to understand how machine learning can be applied to real petroleum engineering problems. Water cut is one of the most important metrics engineers track in an oil well, since it tells you how much of the produced fluid is water versus oil. Predicting it early helps with planning well interventions and forecasting how a well will perform over time.

This started as a simple data generation script and grew into a full pipeline: generating realistic well data, training and comparing two different ML models, evaluating their accuracy, visualizing the results, and finally wrapping everything into an interactive web app so anyone can test it without touching the code.

## 🎯 What It Does

- Generates a synthetic dataset of 300 oil well samples with realistic parameter ranges
- Trains two models, Linear Regression and Random Forest, and compares their performance
- Evaluates both using Mean Absolute Error and R² Score
- Visualizes actual vs predicted water cut, along with residual errors

The web app is organized into two parts:

1. **Model Performance** — shown automatically when the app loads, displaying the R² score, MAE, an actual vs predicted comparison plot, and a residual error plot, so anyone visiting can immediately see how reliable the model is.
2. **Try It Yourself** — an interactive section where you enter your own well parameters and instantly get a predicted water cut, displayed as a pie chart breakdown of water vs oil.

## 🛠️ Built With

1, Python
2, Pandas & NumPy for data handling
3, Scikit-learn for the ML models
4, Matplotlib for visualizations
5, Streamlit for the web app

## 📊 Input Parameters

| Parameter | Description | Unit |
|-----------|--------------|------|
| Oil Rate | Oil production rate | STB/day |
| Gas Rate | Gas production rate | Mscf/day |
| Pressure | Reservoir pressure | psi |
| Temperature | Well temperature | °F |
| Time | Days since well startup | days |

## 🚀 How to Run It

Install the requirements:
```bash
pip install -r requirements.txt
```

Run the web app:
```bash
streamlit run app.py
```

Or run the simpler terminal version:
```bash
python pythonproject.py
```

## 📈 Results

| Model | R² Score | MAE |
|-------|----------|-----|
| Linear Regression | 0.61 | 4.42% |
| Random Forest | 0.57 | 4.80% |

Interestingly, Linear Regression slightly outperformed Random Forest here. That makes sense once you think about it, the underlying water cut formula I used to generate the data was itself linear, so a linear model was naturally a better fit. It was a good reminder that more complex models aren't always better, it depends on the actual relationship in the data.

## 🌐 Live Demo

🔗 *Add your Streamlit Cloud link here once deployed*

## 🔮 What I'd Like to Add Next

- Train on real well production data instead of synthetic data
- Try a few more models like XGBoost and compare
- Add a feature to upload your own CSV dataset
- Track predictions over time for a given well

## 👤 About Me

My name is Atta Ur Rehman. This project focuses on water cut prediction, which is an important problem in the petroleum industry.

I researched different topics related to water cut and developed this project to improve my understanding of petroleum engineering concepts. I have build this with the help of Ai and have knowledge of Python, which helped me understand and work on this project.

This project helped me strengthen my skills, gain practical knowledge, and learn more about solving real-world petroleum industry challenges.

Feel free to connect with me or reach out if you have feedback or suggestions on how to improve it.
