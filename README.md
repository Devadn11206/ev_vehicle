🧾 Problem Statement

Electric Vehicle (EV) users often face range anxiety — uncertainty about how far their vehicle can travel before requiring a recharge.
Existing navigation and energy estimation systems provide only basic, static predictions, failing to consider real-world factors such as driving behavior, terrain, temperature, traffic, and weather conditions.

Additionally, EV drivers lack an intelligent assistant that can:

Explain these range predictions

Suggest energy-efficient driving tips

Plan optimal routes with charging stops

This limits the overall driving experience and reduces trust in EV technology.

Therefore, there is a strong need for a software-based intelligent system that combines:

🧠 Machine Learning for accurate range prediction

💬 Generative AI (GPT) for interactive communication, report generation, and smart route guidance

📊 Dataset

Source: EV Energy Consumption Dataset – Kaggle

Feature	Description
speed	Vehicle speed during trip
distance	Distance covered
battery_energy_used_kwh	Energy consumed
ambient_temp	Outside temperature
...	Additional dataset columns
🧩 Methodology

Data Preprocessing – Handle missing values, normalize features, and derive speed profiles

Feature Engineering – Create new predictors like average speed, elevation gain, and weather factor

Model Training – Train regression models such as XGBoost or LightGBM

Evaluation – Evaluate models using MAE, RMSE, and prediction accuracy

Future Work – Integrate GPT assistant for explanations and optimized route guidance

🧠 Tech Stack

Python

Pandas, NumPy

Scikit-learn / LightGBM

Matplotlib / Seaborn

OpenAI GPT API (future enhancement)

FastAPI / Streamlit (optional UI)

📈 Results
Model	RMSE	MAE
Linear Regression	0.42	0.31
XGBoost	0.27	0.21
🧭 Future Enhancements

Integration with OpenChargeMap API for live charger data

GPT-powered driving and range assistant

Personalized prediction based on driver history

⚙️ How to Run
git clone https://github.com/<your-username>/EV-Range-Prediction-AI.git
cd EV-Range-Prediction-AI
pip install -r requirements.txt
python src/train_model.py