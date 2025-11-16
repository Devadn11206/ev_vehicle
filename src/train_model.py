import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

try:
	from xgboost import XGBRegressor
except Exception as e:
	XGBRegressor = None


EXPECTED_FEATURES = [
	'Vehicle_ID', 'Speed_kmh', 'Acceleration_ms2', 'Battery_State_%',
	'Battery_Voltage_V', 'Battery_Temperature_C', 'Driving_Mode', 'Road_Type',
	'Traffic_Condition', 'Slope_%', 'Weather_Condition', 'Temperature_C',
	'Humidity_%', 'Wind_Speed_ms', 'Tire_Pressure_psi', 'Vehicle_Weight_kg',
	'Distance_Travelled_km', 'Battery_Drop_Rate', 'Temp_Impact',
	'hour', 'dayofweek', 'month'
]


def ensure_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
	# Timestamp features
	if 'Timestamp' in df.columns:
		ts = pd.to_datetime(df['Timestamp'], errors='coerce')
		df['hour'] = ts.dt.hour.fillna(0).astype(int)
		df['dayofweek'] = ts.dt.dayofweek.fillna(0).astype(int)
		df['month'] = ts.dt.month.fillna(1).astype(int)
	else:
		for c, v in [('hour', 0), ('dayofweek', 0), ('month', 1)]:
			if c not in df.columns:
				df[c] = v

	# Energy_per_km
	if 'Energy_per_km' not in df.columns and {'Energy_Consumption_kWh', 'Distance_Travelled_km'}.issubset(df.columns):
		df['Energy_per_km'] = df['Energy_Consumption_kWh'] / (df['Distance_Travelled_km'].replace(0, 1e-6))

	# Battery_Drop_Rate
	if 'Battery_Drop_Rate' not in df.columns and {'Battery_State_%', 'Distance_Travelled_km'}.issubset(df.columns):
		df['Battery_Drop_Rate'] = (100 - df['Battery_State_%']) / (df['Distance_Travelled_km'].replace(0, 1e-6))

	# Temp_Impact
	if 'Temp_Impact' not in df.columns and {'Temperature_C', 'Battery_Temperature_C'}.issubset(df.columns):
		df['Temp_Impact'] = (df['Temperature_C'] - df['Battery_Temperature_C']).abs()

	# Ensure all expected features exist (fill with NaN if missing)
	for c in EXPECTED_FEATURES:
		if c not in df.columns:
			df[c] = np.nan
	return df


def _pick_first_existing(paths):
	for p in paths:
		if os.path.exists(p):
			return p
	return None


def train_model(data_path: str | None, model_path: str):
	if XGBRegressor is None:
		raise RuntimeError("xgboost is not installed. Please install it (pip install xgboost).")

	# Allow finding data in multiple locations
	if data_path is None or not os.path.exists(data_path):
		candidates = [
			os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_ev_data.csv')),
			os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cleaned_ev_data.csv')),
			os.path.abspath(os.path.join(os.getcwd(), 'data', 'cleaned_ev_data.csv')),
		]
		data_path = _pick_first_existing(candidates)
		if data_path is None:
			raise FileNotFoundError("Training data not found in expected locations. Run preprocess.py or provide --data path.")

	df = pd.read_csv(data_path)

	# Basic NA handling and feature engineering
	df = df.dropna(subset=['Energy_Consumption_kWh']).copy()
	df = ensure_engineered_features(df)
	# Ensure Energy_per_km exists as new target
	if 'Energy_per_km' not in df.columns:
		if {'Energy_Consumption_kWh', 'Distance_Travelled_km'}.issubset(df.columns):
			df['Energy_per_km'] = df['Energy_Consumption_kWh'] / df['Distance_Travelled_km'].replace(0, 1e-6)
		else:
			df['Energy_per_km'] = np.nan
	df = df.dropna(subset=['Energy_per_km']).copy()

	# Cast object columns to category codes for XGBoost
	for col in df.columns:
		if df[col].dtype == 'object':
			df[col] = df[col].astype('category')

	# Features exclude target to avoid leakage
	X = df[EXPECTED_FEATURES]
	y = df['Energy_per_km']

	# Train/test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	model = XGBRegressor(
		n_estimators=200,
		learning_rate=0.05,
		max_depth=6,
		subsample=0.8,
		colsample_bytree=0.8,
		random_state=42,
		enable_categorical=True,
		n_jobs=4,
		tree_method='hist'
	)
	model.fit(X_train, y_train)

	# Evaluate
	preds = model.predict(X_test)
	mae = mean_absolute_error(y_test, preds)
	# Compute RMSE in a way compatible with older sklearn versions
	rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
	r2 = r2_score(y_test, preds)
	print(f"(Energy_per_km target) MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

	# Save model
	os.makedirs(os.path.dirname(model_path), exist_ok=True)
	joblib.dump(model, model_path)
	print(f"Model saved to: {model_path}")
	# Save metadata (features + target) for inference alignment
	meta = {
		'features': EXPECTED_FEATURES,
		'target': 'Energy_per_km'
	}
	joblib.dump(meta, model_path + '.meta')
	print(f"Metadata saved to: {model_path}.meta")
	# Also save a copy in repo root data for convenience if different
	alt_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ev_range_model.pkl'))
	try:
		if os.path.abspath(model_path) != alt_model_path:
			os.makedirs(os.path.dirname(alt_model_path), exist_ok=True)
			joblib.dump(model, alt_model_path)
			print(f"Model also saved to: {alt_model_path}")
			joblib.dump(meta, alt_model_path + '.meta')
			print(f"Metadata also saved to: {alt_model_path}.meta")
	except Exception:
		pass


if __name__ == '__main__':
	# Prefer repo-relative paths; allow auto-discovery
	preferred_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_ev_data.csv'))
	model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ev_range_model.pkl'))
	train_model(preferred_data if os.path.exists(preferred_data) else None, model_path)
