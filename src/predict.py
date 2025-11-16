import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import joblib


# Model path candidates (nested and repo root data folders)
MODEL_PATHS = [
	os.path.join(os.path.dirname(__file__), '..', 'data', 'ev_range_model.pkl'),
	os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ev_range_model.pkl'),
	'data/ev_range_model.pkl',
]

# Dataset candidates to derive sensible input ranges
DATA_PATHS = [
	os.path.join(os.path.dirname(__file__), '..', 'data', 'ev_data.csv'),
	os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ev_data.csv'),
	'data/ev_data.csv',
	os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_ev_data.csv'),
	os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cleaned_ev_data.csv'),
	'data/cleaned_ev_data.csv',
]


EXPECTED_FEATURES = [
	'Vehicle_ID', 'Speed_kmh', 'Acceleration_ms2', 'Battery_State_%',
	'Battery_Voltage_V', 'Battery_Temperature_C', 'Driving_Mode', 'Road_Type',
	'Traffic_Condition', 'Slope_%', 'Weather_Condition', 'Temperature_C',
	'Humidity_%', 'Wind_Speed_ms', 'Tire_Pressure_psi', 'Vehicle_Weight_kg',
	'Distance_Travelled_km', 'Energy_per_km', 'Battery_Drop_Rate', 'Temp_Impact',
	'hour', 'dayofweek', 'month'
]


def pick_first_existing(paths):
	for p in paths:
		if os.path.exists(p):
			return p
	return None


def load_model():
	mp = pick_first_existing([os.path.abspath(p) for p in MODEL_PATHS])
	if mp is None:
		raise FileNotFoundError("Model file not found. Train the model first to create data/ev_range_model.pkl.")
	model = joblib.load(mp)
	print("Model loaded successfully!")
	return model


def load_dataset():
	dp = pick_first_existing([os.path.abspath(p) for p in DATA_PATHS])
	if dp:
		try:
			return pd.read_csv(dp)
		except Exception:
			return None
	return None


def qbounds(data_df, col, fallback=(0.0, 100.0, 50.0), as_int=False):
	if data_df is not None and col in data_df.columns:
		s = pd.to_numeric(data_df[col], errors='coerce').dropna()
		if len(s) > 0:
			lo = float(s.quantile(0.05))
			hi = float(s.quantile(0.95))
			med = float(s.quantile(0.5))
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


def prompt_float(label, lo, hi, default, step=1.0):
	while True:
		try:
			raw = input(f"{label} [{lo}..{hi}] (default {default}): ").strip()
			if raw == '':
				return float(default)
			val = float(raw)
			if val < lo:
				print(f"Value below {lo}, clamping.")
				val = lo
			if val > hi:
				print(f"Value above {hi}, clamping.")
				val = hi
			return val
		except Exception:
			print("Please enter a number.")


def prompt_int(label, lo, hi, default):
	while True:
		try:
			raw = input(f"{label} [{lo}..{hi}] (default {default}): ").strip()
			if raw == '':
				return int(default)
			val = int(raw)
			if val < lo:
				print(f"Value below {lo}, clamping.")
				val = lo
			if val > hi:
				print(f"Value above {hi}, clamping.")
				val = hi
			return val
		except Exception:
			print("Please enter an integer.")


def build_feature_row(values: dict) -> pd.DataFrame:
	# Timestamp-derived features
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
		'Energy_per_km': np.nan,  # not known at prediction time
		'Battery_Drop_Rate': ((100 - batt_pct) / (distance + 1e-3)) if distance is not None else np.nan,
		'Temp_Impact': (abs(temp - batt_temp) if (temp is not None and batt_temp is not None) else np.nan),
		'hour': int(ts.hour),
		'dayofweek': int(ts.dayofweek),
		'month': int(ts.month),
	})
	df = pd.DataFrame([row])
	return df[EXPECTED_FEATURES]


def _scale_like_training(features_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
	if ref_df is None:
		return features_df
	ref = ref_df.copy()
	# Work only with expected features to avoid accidental extra columns
	ref = ref[[c for c in EXPECTED_FEATURES if c in ref.columns]].copy()
	scaled = features_df.copy()
	# Numeric columns - mirror preprocess: everything numeric except target-like columns
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
			# If no variation, map to 0.0
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


def predict_trip(values):
	model = load_model()
	features_df = build_feature_row(values)
	# Scale inputs to match training preprocessing (min-max on numeric columns)
	ref_df = load_dataset()
	features_df_scaled = _scale_like_training(features_df, ref_df)
	pred_kwh = float(model.predict(features_df_scaled)[0])  # model predicts total Energy_Consumption_KWh

	distance_km = max(0.0, float(values.get('Distance_Travelled_km', 0.0) or 0.0))
	energy_per_km = pred_kwh / max(distance_km, 1e-3) if distance_km > 0 else pred_kwh

	battery_capacity_kwh = 75.0
	usable_kwh = battery_capacity_kwh * (float(values.get('Battery_State_%', 80)) / 100.0)
	est_range_km = (usable_kwh / max(energy_per_km, 1e-6))

	print(f"Predicted total energy for this trip: {pred_kwh:.3f} kWh")
	print(f"Energy intensity: {energy_per_km:.4f} kWh/km")
	print(f"Estimated max range at current SoC: {est_range_km:.1f} km")


def main():
	data_df = load_dataset()
	print("Enter trip details (press Enter to accept defaults):")

	sp_lo, sp_hi, sp_med = qbounds(data_df, 'Speed_kmh', fallback=(0, 200, 60), as_int=True)
	speed = prompt_int("Speed (km/h)", sp_lo, sp_hi, sp_med)

	acc_lo, acc_hi, acc_med = qbounds(data_df, 'Acceleration_ms2', fallback=(0.0, 5.0, 1.2), as_int=False)
	acceleration = prompt_float("Acceleration (m/s^2)", acc_lo, acc_hi, acc_med, step=0.1)

	t_lo, t_hi, t_med = qbounds(data_df, 'Temperature_C', fallback=(-30, 60, 30), as_int=True)
	temperature = prompt_int("Temperature (°C)", t_lo, t_hi, t_med)

	h_lo, h_hi, h_med = qbounds(data_df, 'Humidity_%', fallback=(0, 100, 60), as_int=True)
	humidity = prompt_int("Humidity (%)", h_lo, h_hi, h_med)

	s_lo, s_hi, s_med = qbounds(data_df, 'Slope_%', fallback=(-10, 10, 1), as_int=True)
	slope = prompt_int("Road Slope (%)", s_lo, s_hi, s_med)

	b_lo, b_hi, b_med = qbounds(data_df, 'Battery_State_%', fallback=(0, 100, 80), as_int=True)
	battery_state = prompt_int("Battery State (%)", b_lo, b_hi, b_med)

	d_lo, d_hi, d_med = qbounds(data_df, 'Distance_Travelled_km', fallback=(1, 2000, 20), as_int=True)
	distance = prompt_int("Distance Travelled (km)", max(1, d_lo), d_hi, max(1, d_med))

	tp_lo, tp_hi, tp_med = qbounds(data_df, 'Tire_Pressure_psi', fallback=(25, 50, 35), as_int=True)
	tire_pressure = prompt_int("Tire Pressure (psi)", tp_lo, tp_hi, tp_med)

	w_lo, w_hi, w_med = qbounds(data_df, 'Wind_Speed_ms', fallback=(0, 60, 5), as_int=False)
	wind_speed = prompt_float("Wind Speed (m/s)", w_lo, w_hi, w_med, step=0.5)

	# Advanced defaults from fallbacks/medians
	v_lo, v_hi, v_med = qbounds(data_df, 'Battery_Voltage_V', fallback=(300, 900, 350), as_int=True)
	battery_voltage = v_med
	bt_lo, bt_hi, bt_med = qbounds(data_df, 'Battery_Temperature_C', fallback=(-30, 80, 30), as_int=True)
	battery_temp = bt_med
	vw_lo, vw_hi, vw_med = qbounds(data_df, 'Vehicle_Weight_kg', fallback=(800, 3000, 1500), as_int=True)
	vehicle_weight = vw_med

	ts = input("Timestamp (YYYY-MM-DD HH:MM, optional; Enter for now): ").strip()
	timestamp = ts if ts else None

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
		"Timestamp": timestamp,
	}

	predict_trip(values)


if __name__ == "__main__":
	if sys.stdin.isatty():
		main()
	else:
		# Non-interactive fallback
		predict_trip({
			"Speed_kmh": 80,
			"Acceleration_ms2": 0.6,
			"Battery_State_%": 85,
			"Battery_Voltage_V": 350,
			"Battery_Temperature_C": 30,
			"Driving_Mode": 1,
			"Road_Type": 0,
			"Traffic_Condition": 1,
			"Slope_%": 1,
			"Weather_Condition": 1,
			"Temperature_C": 30,
			"Humidity_%": 60,
			"Wind_Speed_ms": 3,
			"Tire_Pressure_psi": 34,
			"Vehicle_Weight_kg": 1500,
			"Distance_Travelled_km": 50,
			"Timestamp": str(datetime.now()),
		})

