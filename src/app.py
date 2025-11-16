import os
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

import streamlit as st

# Try optional imports
try:
	import matplotlib.pyplot as plt
except Exception:
	plt = None

# --------------------
# Shared definitions
# --------------------

EXPECTED_FEATURES = [
	'Vehicle_ID', 'Speed_kmh', 'Acceleration_ms2', 'Battery_State_%',
	'Battery_Voltage_V', 'Battery_Temperature_C', 'Driving_Mode', 'Road_Type',
	'Traffic_Condition', 'Slope_%', 'Weather_Condition', 'Temperature_C',
	'Humidity_%', 'Wind_Speed_ms', 'Tire_Pressure_psi', 'Vehicle_Weight_kg',
	'Distance_Travelled_km', 'Energy_per_km', 'Battery_Drop_Rate', 'Temp_Impact',
	'hour', 'dayofweek', 'month'
]

MODEL_PATHS = [
	os.path.join(os.path.dirname(__file__), '..', 'data', 'ev_range_model.pkl'),
	os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ev_range_model.pkl'),
	'data/ev_range_model.pkl',
]

DATA_PATHS = [
	os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_ev_data.csv'),
	os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cleaned_ev_data.csv'),
	'data/cleaned_ev_data.csv',
	os.path.join(os.path.dirname(__file__), '..', 'data', 'ev_data.csv'),
	os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ev_data.csv'),
	'data/ev_data.csv',
]


def _pick_first_existing(paths):
	for p in paths:
		ap = os.path.abspath(p)
		if os.path.exists(ap):
			return ap
	return None


@st.cache_resource(show_spinner=False)
def load_model():
	mp = _pick_first_existing(MODEL_PATHS)
	if mp is None:
		raise FileNotFoundError("Model file not found. Train the model first to create data/ev_range_model.pkl.")
	return joblib.load(mp)


@st.cache_data(show_spinner=False)
def load_dataset():
	dp = _pick_first_existing(DATA_PATHS)
	if dp is None:
		return None
	try:
		return pd.read_csv(dp)
	except Exception:
		return None


def qbounds(data_df, col, fallback=(0.0, 100.0, 50.0), as_int=False):
	if data_df is not None and col in data_df.columns:
		s = pd.to_numeric(data_df[col], errors='coerce').dropna()
		if len(s) > 0:
			lo = float(s.quantile(0.05))
			hi = float(s.quantile(0.95))
			med = float(s.quantile(0.5))
			# Detect if column looks normalized [0,1] and prefer domain fallback
			likely_norm = (hi <= 1.2) and (hi - lo <= 1.2)
			if likely_norm and col in {
				'Speed_kmh','Acceleration_ms2','Temperature_C','Humidity_%','Slope_%','Battery_State_%',
				'Distance_Travelled_km','Tire_Pressure_psi','Wind_Speed_ms','Battery_Voltage_V',
				'Battery_Temperature_C','Vehicle_Weight_kg'
			}:
				lo, hi, med = fallback
			if as_int:
				return int(np.floor(lo)), int(np.ceil(hi)), int(round(med))
			return lo, hi, med
	lo, hi, med = fallback
	if as_int:
		return int(lo), int(hi), int(med)
	return float(lo), float(hi), float(med)


def _scale_like_training(features_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
	if ref_df is None:
		return features_df
	ref = ref_df.copy()
	ref = ref[[c for c in EXPECTED_FEATURES if c in ref.columns]].copy()
	scaled = features_df.copy()
	numeric_cols = ref.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
	exclude = {'Energy_Consumption_kWh', 'Energy_per_km'}
	for col in numeric_cols:
		if col in exclude or col not in scaled.columns:
			continue
		ref_col = pd.to_numeric(ref[col], errors='coerce')
		cmin = ref_col.min()
		cmax = ref_col.max()
		denom = (cmax - cmin) if pd.notna(cmax) and pd.notna(cmin) and (cmax != cmin) else 0.0
		if denom == 0.0:
			try:
				scaled[col] = 0.0
			except Exception:
				pass
		else:
			try:
				scaled[col] = (pd.to_numeric(scaled[col], errors='coerce') - cmin) / denom
			except Exception:
				pass
	return scaled


def build_feature_row(values: dict) -> pd.DataFrame:
	ts_input = values.get('Timestamp')
	if ts_input:
		try:
			ts = pd.to_datetime(ts_input)
		except Exception:
			ts = pd.Timestamp(datetime.now())
	else:
		ts = pd.Timestamp(datetime.now())

	distance = values.get('Distance_Travelled_km', np.nan)
	batt_pct = values.get('Battery_State_%', np.nan)
	temp = values.get('Temperature_C', np.nan)
	batt_temp = values.get('Battery_Temperature_C', np.nan)

	row = {k: np.nan for k in EXPECTED_FEATURES}
	row.update({
		'Vehicle_ID': values.get('Vehicle_ID', 0),
		'Speed_kmh': values.get('Speed_kmh'),
		'Acceleration_ms2': values.get('Acceleration_ms2'),
		'Battery_State_%': batt_pct,
		'Battery_Voltage_V': values.get('Battery_Voltage_V', np.nan),
		'Battery_Temperature_C': batt_temp if batt_temp is not None else np.nan,
		'Driving_Mode': values.get('Driving_Mode', 1),
		'Road_Type': values.get('Road_Type', 0),
		'Traffic_Condition': values.get('Traffic_Condition', 1),
		'Slope_%': values.get('Slope_%', 0),
		'Weather_Condition': values.get('Weather_Condition', 1),
		'Temperature_C': temp,
		'Humidity_%': values.get('Humidity_%'),
		'Wind_Speed_ms': values.get('Wind_Speed_ms'),
		'Tire_Pressure_psi': values.get('Tire_Pressure_psi'),
		'Vehicle_Weight_kg': values.get('Vehicle_Weight_kg', 1500),
		'Distance_Travelled_km': distance,
		'Energy_per_km': np.nan,
		'Battery_Drop_Rate': ((100 - batt_pct) / (distance + 1e-3)) if distance is not None else np.nan,
		'Temp_Impact': (abs(temp - batt_temp) if (temp is not None and batt_temp is not None) else np.nan),
		'hour': int(ts.hour),
		'dayofweek': int(ts.dayofweek),
		'month': int(ts.month),
	})
	df = pd.DataFrame([row])
	return df[EXPECTED_FEATURES]


def predict_energy_and_range(model, values: dict, ref_df: pd.DataFrame):
	features_df = build_feature_row(values)
	features_df_scaled = _scale_like_training(features_df, ref_df)
	pred_kwh = float(model.predict(features_df_scaled)[0])
	distance_km = max(0.0, float(values.get('Distance_Travelled_km', 0.0) or 0.0))
	energy_per_km = pred_kwh / max(distance_km, 1e-3) if distance_km > 0 else pred_kwh
	battery_capacity_kwh = float(values.get('Battery_Capacity_kWh', 75.0))
	usable_kwh = battery_capacity_kwh * (float(values.get('Battery_State_%', 80)) / 100.0)
	est_range_km = (usable_kwh / max(energy_per_km, 1e-6))
	return pred_kwh, energy_per_km, est_range_km


def page_predict():
	st.header("EV Range Prediction")
	data_df = load_dataset()

	# Reasonable bounds from data or fallbacks
	sp_lo, sp_hi, sp_med = qbounds(data_df, 'Speed_kmh', fallback=(0, 200, 60), as_int=True)
	acc_lo, acc_hi, acc_med = qbounds(data_df, 'Acceleration_ms2', fallback=(0.0, 5.0, 1.2), as_int=False)
	t_lo, t_hi, t_med = qbounds(data_df, 'Temperature_C', fallback=(-30, 60, 25), as_int=True)
	h_lo, h_hi, h_med = qbounds(data_df, 'Humidity_%', fallback=(0, 100, 60), as_int=True)
	s_lo, s_hi, s_med = qbounds(data_df, 'Slope_%', fallback=(-10, 10, 1), as_int=True)
	b_lo, b_hi, b_med = qbounds(data_df, 'Battery_State_%', fallback=(0, 100, 80), as_int=True)
	d_lo, d_hi, d_med = qbounds(data_df, 'Distance_Travelled_km', fallback=(1, 2000, 20), as_int=True)
	tp_lo, tp_hi, tp_med = qbounds(data_df, 'Tire_Pressure_psi', fallback=(25, 50, 35), as_int=True)
	w_lo, w_hi, w_med = qbounds(data_df, 'Wind_Speed_ms', fallback=(0, 60, 5), as_int=False)
	v_lo, v_hi, v_med = qbounds(data_df, 'Battery_Voltage_V', fallback=(300, 900, 350), as_int=True)
	bt_lo, bt_hi, bt_med = qbounds(data_df, 'Battery_Temperature_C', fallback=(-30, 80, 30), as_int=True)
	vw_lo, vw_hi, vw_med = qbounds(data_df, 'Vehicle_Weight_kg', fallback=(800, 3000, 1500), as_int=True)

	with st.form("predict_form"):
		c1, c2, c3 = st.columns(3)
		with c1:
			speed = st.number_input("Speed (km/h)", min_value=sp_lo, max_value=sp_hi, value=sp_med)
			temperature = st.number_input("Temperature (°C)", min_value=t_lo, max_value=t_hi, value=t_med)
			slope = st.number_input("Road Slope (%)", min_value=s_lo, max_value=s_hi, value=s_med)
			tire_pressure = st.number_input("Tire Pressure (psi)", min_value=tp_lo, max_value=tp_hi, value=tp_med)
			battery_capacity = st.number_input("Battery Capacity (kWh)", min_value=20.0, max_value=200.0, value=75.0)
		with c2:
			acceleration = st.number_input("Acceleration (m/s²)", min_value=float(acc_lo), max_value=float(acc_hi), value=float(acc_med), step=0.1)
			humidity = st.number_input("Humidity (%)", min_value=h_lo, max_value=h_hi, value=h_med)
			wind_speed = st.number_input("Wind Speed (m/s)", min_value=float(w_lo), max_value=float(w_hi), value=float(w_med), step=0.5)
			battery_voltage = st.number_input("Battery Voltage (V)", min_value=v_lo, max_value=v_hi, value=v_med)
		with c3:
			battery_state = st.number_input("Battery State of Charge (%)", min_value=b_lo, max_value=b_hi, value=b_med)
			battery_temp = st.number_input("Battery Temperature (°C)", min_value=bt_lo, max_value=bt_hi, value=bt_med)
			distance = st.number_input("Trip Distance (km)", min_value=max(1, d_lo), max_value=d_hi, value=max(1, d_med))
			vehicle_weight = st.number_input("Vehicle Weight (kg)", min_value=vw_lo, max_value=vw_hi, value=vw_med)

		timestamp = st.text_input("Timestamp (YYYY-MM-DD HH:MM, optional)", value="")
		submitted = st.form_submit_button("Predict Range")

	if submitted:
		try:
			model = load_model()
		except Exception as e:
			st.error(str(e))
			return

		values = {
			"Speed_kmh": speed,
			"Acceleration_ms2": acceleration,
			"Battery_State_%": battery_state,
			"Battery_Voltage_V": battery_voltage,
			"Battery_Temperature_C": battery_temp,
			"Driving_Mode": 1,
			"Road_Type": 0,
			"Traffic_Condition": 1,
			"Slope_%": slope,
			"Weather_Condition": 1,
			"Temperature_C": temperature,
			"Humidity_%": humidity,
			"Wind_Speed_ms": wind_speed,
			"Tire_Pressure_psi": tire_pressure,
			"Vehicle_Weight_kg": vehicle_weight,
			"Distance_Travelled_km": distance,
			"Timestamp": timestamp if timestamp.strip() else None,
			"Battery_Capacity_kWh": battery_capacity,
		}

		pred_kwh, energy_per_km, est_range_km = predict_energy_and_range(model, values, load_dataset())

		# KPIs
		k1, k2, k3 = st.columns(3)
		k1.metric("Predicted energy", f"{pred_kwh:.3f} kWh")
		k2.metric("Energy intensity", f"{energy_per_km:.4f} kWh/km")
		k3.metric("Estimated range", f"{est_range_km:.1f} km")

		# Simple chart: bar of energy and range
		st.subheader("Trip summary")
		summary_df = pd.DataFrame({
			'Metric': ['Predicted energy (kWh)', 'Energy intensity (kWh/km)', 'Estimated range (km)'],
			'Value': [pred_kwh, energy_per_km, est_range_km]
		})
		st.dataframe(summary_df, use_container_width=True, hide_index=True)


def page_evaluation():
	st.header("Model Evaluation: Actual vs Predicted")
	data_df = load_dataset()
	if data_df is None or 'Energy_Consumption_kWh' not in data_df.columns:
		st.info("Could not find cleaned_ev_data.csv with 'Energy_Consumption_kWh'. Train/preprocess the dataset first.")
		return

	# Import helper to engineer any missing features
	try:
		from train_model import ensure_engineered_features
	except Exception:
		ensure_engineered_features = None

	df = data_df.copy()
	if ensure_engineered_features is not None:
		try:
			df = ensure_engineered_features(df)
		except Exception:
			pass

	# Build X matrix aligned to EXPECTED_FEATURES
	for c in EXPECTED_FEATURES:
		if c not in df.columns:
			df[c] = np.nan
	X = df[EXPECTED_FEATURES].copy()
	y = pd.to_numeric(df['Energy_Consumption_kWh'], errors='coerce')
	mask = y.notna()
	X = X.loc[mask]
	y = y.loc[mask]

	# Cast objects to categories like training
	for col in X.columns:
		if X[col].dtype == 'object':
			X[col] = X[col].astype('category')

	try:
		model = load_model()
	except Exception as e:
		st.error(str(e))
		return

	Xs = _scale_like_training(X, df)
	preds = model.predict(Xs)

	# Metrics
	from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
	mae = float(mean_absolute_error(y, preds))
	rmse = float(np.sqrt(mean_squared_error(y, preds)))
	r2 = float(r2_score(y, preds))
	mape = float(np.mean(np.abs((y - preds) / np.maximum(1e-6, y)))) * 100.0

	k1, k2, k3, k4 = st.columns(4)
	k1.metric("MAE", f"{mae:.3f} kWh")
	k2.metric("RMSE", f"{rmse:.3f} kWh")
	k3.metric("R² (accuracy)", f"{r2*100:.2f}%")
	k4.metric("MAPE", f"{mape:.2f}%")

	st.subheader("Actual vs Predicted (parity plot)")
	if plt is not None:
		fig, ax = plt.subplots(figsize=(6, 6))
		ax.scatter(y, preds, alpha=0.4)
		lo = min(y.min(), preds.min())
		hi = max(y.max(), preds.max())
		ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
		ax.set_xlabel('Actual Energy (kWh)')
		ax.set_ylabel('Predicted Energy (kWh)')
		ax.set_title('Actual vs Predicted')
		st.pyplot(fig)
	else:
		st.line_chart(pd.DataFrame({'Actual': y.values[:200], 'Predicted': preds[:200]}))

	st.caption("Metrics computed on the available cleaned dataset using the current model.")


def page_about():
	st.header("About")
	st.write(
		"This app predicts EV trip energy usage and estimated range using an XGBoost model. "
		"Use the Predict page to enter trip conditions. The Model Evaluation page compares the model's predictions to the dataset."
	)


def main():
	st.set_page_config(page_title="EV Range Prediction", layout="wide")
	st.sidebar.title("Navigation")
	page = st.sidebar.radio("Go to", ["Predict Range", "Model Evaluation", "About"]) 

	if page == "Predict Range":
		page_predict()
	elif page == "Model Evaluation":
		page_evaluation()
	else:
		page_about()


if __name__ == "__main__":
	main()


