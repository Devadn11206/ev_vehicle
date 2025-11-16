import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

def preprocess_data(input_path, output_path):
    # Step 1: Load dataset
    df = pd.read_csv(input_path)
    print(f"✅ Data loaded successfully! Shape: {df.shape}")

    # Step 2: Drop unnecessary or duplicate rows
    df = df.drop_duplicates()
    
    # Step 3: Handle missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Compute Energy_per_km if possible and missing
    if 'Energy_per_km' not in df.columns and {'Energy_Consumption_kWh', 'Distance_Travelled_km'}.issubset(df.columns):
        # Avoid divide by zero
        df['Energy_per_km'] = df['Energy_Consumption_kWh'] / (df['Distance_Travelled_km'].replace(0, 1e-6))

    # Step 4: Encode categorical features
    categorical_cols = ['Driving_Mode', 'Road_Type', 'Traffic_Condition', 'Weather_Condition']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Step 5: Normalize numeric features (exclude target-like columns)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    exclude_from_scaling = {'Energy_Consumption_kWh', 'Energy_per_km'}
    scale_cols = [c for c in numeric_cols if c not in exclude_from_scaling]
    if len(scale_cols) > 0:
        scaler = MinMaxScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # Step 6: Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to: {output_path}")

    return df

if __name__ == "__main__":
    input_file = r"C:\Users\ddnan\Downloads\EV_Energy_Consumption_Dataset.csv"
    output_file = r"C:\Users\ddnan\EV-Range-Prediction-AI\data\cleaned_ev_data.csv"
    preprocess_data(input_file, output_file)
