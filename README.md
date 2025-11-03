# ⚡ EV Range Prediction AI 🚗🔋  
*An AI-powered system to predict the ideal and actual range of Electric Vehicles based on real-world driving conditions.*

---

## 🧠 Project Overview  
This project predicts the **electric vehicle (EV) range** using **machine learning** and **environmental parameters** like speed, temperature, humidity, slope, and battery state.  
It also visualizes **ideal vs predicted range** using interactive charts in a **Streamlit app**.

---

## 🌟 Key Features  
✅ Predicts EV range using trained ML models (e.g., Linear Regression / Random Forest)  
✅ Interactive input fields for real-time prediction  
✅ Visual comparison of *Ideal Range vs Predicted Range*  
✅ Dynamic visualization using bar and line graphs  
✅ Clean, responsive Streamlit interface  
✅ Easily extendable for route optimization or battery efficiency tracking  

---

## ⚙️ Tech Stack  
| Component | Technology Used |
|------------|-----------------|
| Programming Language | Python |
| Frontend | Streamlit |
| ML Libraries | scikit-learn, pandas, numpy |
| Visualization | Matplotlib, Plotly |
| Data Handling | CSV Datasets |

---

## 🧩 Input Parameters  
The model considers the following inputs to estimate EV range:

| Parameter | Description |
|------------|-------------|
| Speed (km/h) | Vehicle’s average speed |
| Acceleration (m/s²) | Rate of increase in speed |
| Temperature (°C) | Ambient temperature |
| Humidity (%) | Moisture in the air |
| Road Slope (%) | Incline or decline percentage |
| Battery State (%) | Remaining battery charge |
| Distance Travelled (km) | Distance already covered |
| Tire Pressure (psi) | Tire air pressure |
| Wind Speed (m/s) | Air resistance factor |

---

## 📊 Example Output  
- Predicted Range: **280 km**  
- Ideal Range: **300 km**  
- Visualized in bar/line chart comparison  

---

## 🚀 How to Run Locally  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Devadn11206/EV-Range-Prediction-AI.git
cd EV-Range-Prediction-AI
2️⃣ Create and Activate Virtual Environment
bash
Copy code
python -m venv .venv
.venv\Scripts\activate      # For Windows
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the Streamlit App
bash
Copy code
streamlit run app.py
💡 Future Enhancements
🔹 Integrate Generative AI chatbot for real-time EV insights
🔹 Add route optimization and charging station suggestions
🔹 Build a REST API for mobile app integration
🔹 Collect real driving datasets for model improvement

🧑‍💻 Author
Devanandu
🚀 Passionate AI | ML | Data Science Learner
📫 GitHub: Devadn11206



---

Would you like me to make this **README.md** look even more *GitHub-styled with emojis, tables, and badges* (like stars, license, or "Made with ❤️ in Python")?  
It’ll make your repo look like a **professional AI portfolio project**.
