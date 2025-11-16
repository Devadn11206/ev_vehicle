import os
import streamlit as st
from groq import Groq
import joblib
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
import re
from datetime import datetime
from pathlib import Path

# Optional .env support
try:  # Lightweight attempt; if python-dotenv not installed we silently skip
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------
MODEL_NAME = "llama-3.1-8b-instant"

# Primary sources: environment variable or Streamlit secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or (st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None)

# Fallback: read from plain text file if still missing
if not GROQ_API_KEY:
    fallback_files = [
        os.path.join(os.getcwd(), "GROQ_API_KEY.txt"),
        os.path.join(os.path.dirname(__file__), "..", "GROQ_API_KEY.txt"),
        os.path.join(os.path.dirname(__file__), "..", "..", "GROQ_API_KEY.txt"),
    ]
    for fp in fallback_files:
        try:
            if os.path.exists(fp):
                with open(fp, "r", encoding="utf-8") as f:
                    candidate = f.read().strip()
                    if candidate:
                        GROQ_API_KEY = candidate
                        break
        except Exception:
            pass


def _mask_key(k: Optional[str]) -> str:
    if not k:
        return "(none)"
    k = k.strip()
    if len(k) <= 12:
        return k
    return f"{k[:6]}...{k[-4:]} (len {len(k)})"

MODEL_CANDIDATES: List[str] = [
    "ev_range_model.pkl",
    os.path.join("data", "ev_range_model.pkl"),
    os.path.join(os.getcwd(), "data", "ev_range_model.pkl"),
    os.path.join(os.path.dirname(__file__), "..", "data", "ev_range_model.pkl"),
]

@st.cache_resource(show_spinner=False)
def load_ml_model():
    for p in MODEL_CANDIDATES:
        candidate = os.path.abspath(p)
        if os.path.exists(candidate):
            try:
                model = joblib.load(candidate)
                # Try to load metadata
                meta_path = candidate + '.meta'
                meta = None
                if os.path.exists(meta_path):
                    try:
                        meta = joblib.load(meta_path)
                    except Exception:
                        meta = None
                return model, meta
            except Exception:
                continue
    return None, None

ml_model, model_meta = load_ml_model()

def init_groq(key: Optional[str] = None):
    """Initialize Groq client without caching so key changes take effect immediately."""
    use_key = key or GROQ_API_KEY
    if not use_key:
        raise RuntimeError("Groq API key missing. Set GROQ_API_KEY env var or add to Streamlit secrets.")
    return Groq(api_key=use_key)

SYSTEM_PROMPT = (
    """You are EV-Genie, an expert Electric Vehicle assistant.
You explain EV range, charging, batteries, efficiency, temperature effects,
and maintenance in simple and helpful ways."""
)

RANGE_KEYWORDS = ["range", "how far", "estimate", "distance", "travel"]

# Prediction cache to avoid recalculating identical parameter sets
_prediction_cache: Dict[Tuple[float, float, float, float], Dict[str, float]] = {}

USAGE_LOG_PATH = Path(os.path.dirname(__file__)) / ".." / "data" / "usage_log.csv"

def _log_row(kind: str, payload: Dict):
    try:
        USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        header_needed = not USAGE_LOG_PATH.exists()
        with USAGE_LOG_PATH.open("a", encoding="utf-8") as f:
            if header_needed:
                f.write("timestamp,kind," + ",".join(sorted(payload.keys())) + "\n")
            row = datetime.utcnow().isoformat() + "," + kind + "," + ",".join(str(payload.get(k, "")) for k in sorted(payload.keys()))
            f.write(row + "\n")
    except Exception:
        pass

def extract_parameters(query: str) -> Dict[str, float]:
    """Parse battery %, capacity (kWh), speed (km/h or mph), temperature (C or F)."""
    params: Dict[str, float] = {}
    # Battery percent
    m = re.search(r"(\d{1,3})\s*%", query)
    if m:
        val = float(m.group(1))
        if 0 <= val <= 100:
            params['battery_percent'] = val
    # Capacity kWh
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*kWh", query, re.IGNORECASE)
    if m:
        params['battery_capacity'] = float(m.group(1))
    # Speed km/h
    m = re.search(r"(\d{1,3})\s*km/?h", query, re.IGNORECASE)
    if m:
        params['speed'] = float(m.group(1))
    # Speed mph -> convert
    m = re.search(r"(\d{1,3})\s*mph", query, re.IGNORECASE)
    if m and 'speed' not in params:
        mph = float(m.group(1))
        params['speed'] = round(mph * 1.60934, 2)
    # Temperature C
    m = re.search(r"(-?\d{1,2})\s*°?C", query, re.IGNORECASE)
    if m:
        params['temp'] = float(m.group(1))
    # Temperature F -> convert
    m = re.search(r"(-?\d{1,3})\s*°?F", query, re.IGNORECASE)
    if m and 'temp' not in params:
        f = float(m.group(1))
        params['temp'] = round((f - 32) * 5/9, 2)
    return params

def can_estimate_range(query: str) -> bool:
    q_low = query.lower()
    return any(k in q_low for k in RANGE_KEYWORDS)

def predict_range_with_model(params: Dict[str, float]) -> Optional[Dict[str, float]]:
    if ml_model is None:
        return None
    required = {'battery_percent', 'battery_capacity', 'speed', 'temp'}
    if not required.issubset(params):
        return None
    battery_percent = params['battery_percent']
    battery_capacity = params['battery_capacity']
    speed = params['speed']
    temp = params['temp']
    cache_key = (battery_percent, battery_capacity, speed, temp)
    if cache_key in _prediction_cache:
        return _prediction_cache[cache_key]
    base_row = {
        'Vehicle_ID': 1,
        'Speed_kmh': speed,
        'Acceleration_ms2': 0,
        'Battery_State_%': battery_percent,
        'Battery_Voltage_V': 350,
        'Battery_Temperature_C': temp,
        'Driving_Mode': "normal",
        'Road_Type': "city",
        'Traffic_Condition': "moderate",
        'Slope_%': 0,
        'Weather_Condition': "clear",
        'Temperature_C': temp,
        'Humidity_%': 50,
        'Wind_Speed_ms': 1,
        'Tire_Pressure_psi': 32,
        'Vehicle_Weight_kg': 1500,
        'Distance_Travelled_km': 1,
        'Energy_per_km': np.nan,
        'Battery_Drop_Rate': np.nan,
        'Temp_Impact': abs(temp - temp),
        'hour': 12,
        'dayofweek': 3,
        'month': 6,
    }
    df = pd.DataFrame([base_row])
    expected_features = [
        'Vehicle_ID','Speed_kmh','Acceleration_ms2','Battery_State_%','Battery_Voltage_V','Battery_Temperature_C',
        'Driving_Mode','Road_Type','Traffic_Condition','Slope_%','Weather_Condition','Temperature_C','Humidity_%','Wind_Speed_ms',
        'Tire_Pressure_psi','Vehicle_Weight_kg','Distance_Travelled_km','Battery_Drop_Rate','Temp_Impact','hour','dayofweek','month'
    ]
    for col in expected_features:
        if col not in df.columns:
            df[col] = np.nan
    df = df[expected_features]
    try:
        # Cast object/string columns to categorical to match training
        categorical_cols = ['Driving_Mode','Road_Type','Traffic_Condition','Weather_Condition']
        for c in categorical_cols:
            if c in df.columns and df[c].dtype == object:
                df[c] = df[c].astype('category')
        kwh_per_km = float(ml_model.predict(df)[0])
        if not np.isfinite(kwh_per_km) or kwh_per_km <= 0:
            return None
        available_kwh = battery_capacity * (battery_percent / 100.0)
        estimated_range = available_kwh / kwh_per_km
        result = {
            'kwh_per_km': kwh_per_km,
            'estimated_range_km': estimated_range,
            'available_kwh': available_kwh,
        }
        _prediction_cache[cache_key] = result
        _log_row("prediction", {
            'battery_percent': battery_percent,
            'battery_capacity': battery_capacity,
            'speed_kmh': speed,
            'temp_c': temp,
            'kwh_per_km': kwh_per_km,
            'estimated_range_km': estimated_range
        })
        return result
    except Exception:
        return None

def _trim_history(history: List[Tuple[str, str]], max_pairs: int = 6) -> List[Tuple[str, str]]:
    if len(history) <= max_pairs * 2:
        return history
    return history[-max_pairs * 2:]

def call_groq_chat(query: str, history: Optional[List[Tuple[str, str]]] = None, stream: bool = False) -> str:
    if not query.strip():
        return "⚠️ Empty query provided."
    try:
        client = init_groq()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for role, msg in _trim_history(history):
                messages.append({"role": role, "content": msg})
        ml_context = ""
        params = extract_parameters(query)
        if can_estimate_range(query):
            prediction = predict_range_with_model(params)
            if prediction is None:
                if ml_model is None:
                    ml_context = "(Range intent detected, but ML model not loaded. Provide battery %, capacity kWh, speed km/h, temp C for estimate.)"
                else:
                    missing = {k for k in ['battery_percent','battery_capacity','speed','temp'] if k not in params}
                    ml_context = f"(Range intent detected. Missing parameters: {', '.join(sorted(missing))}.)" if missing else "(Unable to compute range with provided values.)"
            else:
                ml_context = (
                    f"(Model estimate: ~{prediction['estimated_range_km']:.1f} km range. Consumption {prediction['kwh_per_km']:.3f} kWh/km from {prediction['available_kwh']:.1f} kWh available.)"
                )
        user_content = query if not ml_context else f"{query}\n\n{ml_context}"
        messages.append({"role": "user", "content": user_content})
        attempts = 3
        last_err: Optional[Exception] = None
        if stream:
            for i in range(attempts):
                try:
                    response_stream = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=512,
                        temperature=0.2,
                        stream=True,
                    )
                    collected = []
                    for chunk in response_stream:
                        part = getattr(chunk.choices[0].delta, "content", "") or ""
                        if part:
                            collected.append(part)
                    return "".join(collected).strip() or "(No content returned)"
                except Exception as e:
                    last_err = e
                    # Only retry on transient network related issues
                    msg = str(e).lower()
                    if "timeout" in msg or "connection" in msg or "rate" in msg or "429" in msg:
                        import time
                        time.sleep(1.5 * (i + 1))
                        continue
                    break
        else:
            for i in range(attempts):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=512,
                        temperature=0.2,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    if "timeout" in msg or "connection" in msg or "rate" in msg or "429" in msg:
                        import time
                        time.sleep(1.5 * (i + 1))
                        continue
                    break
        raise last_err if last_err else RuntimeError("Unknown LLM failure")
    except Exception as e:
        emsg = str(e)
        low = emsg.lower()
        if "invalid api key" in low or "invalid_api_key" in low or "401" in low:
            return "⚠️ Error contacting model: Invalid API key. Re-copy full key (starts with gsk_), set `$env:GROQ_API_KEY`, or use sidebar override."
        return f"⚠️ Error contacting model: {emsg}" 

def run_app():
    st.title("⚡ EV-Genie — Smart EV Assistant")
    st.caption("Reliable EV Q&A + ML range estimator (integrated ML predictions)")
    # Sidebar: API key diagnostics
    with st.sidebar:
        st.markdown("### 🔑 Groq API Key")
        override_key = st.text_input("Override key (not stored)", value="" , type="password")
        # Sanitize override (remove surrounding quotes & whitespace)
        cleaned_override = override_key.strip().strip('"').strip("'").replace("\r"," ").replace("\n"," ")
        active_key = cleaned_override or GROQ_API_KEY
        st.write(f"Active: {_mask_key(active_key)}")
        if active_key:
            # Basic format diagnostics
            if not active_key.startswith("gsk_"):
                st.warning("Key does not start with 'gsk_'. Verify you copied the full Groq key.")
            if len(active_key.strip()) < 32:
                st.warning("Key seems shorter than expected; may be truncated.")
            try:
                client = init_groq(active_key)
                client.chat.completions.create(model=MODEL_NAME, messages=[{"role":"user","content":"ping"}], max_tokens=1, temperature=0)
                st.success("Key valid")
            except Exception as e:
                msg = str(e)
                lowered = msg.lower()
                if "invalid_api_key" in lowered or "401" in lowered:
                    st.error("Invalid API key. Troubleshooting: ensure full key (no trailing spaces), restart app after environment changes, or use override box.")
                    st.caption("Set persistently: run `setx GROQ_API_KEY \"your_full_key\"` then open NEW shell. For .env create file with GROQ_API_KEY=... and restart.")
                else:
                    st.warning(f"Key issue: {msg[:160]}")
        else:
            st.error("No key detected.")
            st.caption("Set env var `GROQ_API_KEY` or paste into override box. Optional: use a .env file.")
        st.caption("Enter a temporary key above or set environment variable.")
        with st.expander("Key Debug / Source"):
            source = "override" if cleaned_override else ("env/secrets" if GROQ_API_KEY else "file fallback" if active_key else "none")
            st.code(f"Source: {source}\nLength: {len(active_key) if active_key else 0}")
            if active_key:
                invisible = [c for c in active_key if ord(c) < 32]
                if invisible:
                    st.warning(f"Key contains control characters: {invisible}. Re-copy plain text key.")
            st.caption("If issues persist: create `.env` with GROQ_API_KEY, restart new shell, or use GROQ_API_KEY.txt fallback.")
    tab1, tab2 = st.tabs(["💬 EV Chatbot", "🔋 EV Range Prediction"])
    with tab1:
        st.header("💬 Ask EV-Genie anything!")
        if "history" not in st.session_state:
            st.session_state.history = []
        user_input = st.text_input("Your question:", placeholder="e.g., Estimate range 70% 40kWh at 60km/h 25C")
        stream_answer = st.checkbox("Stream answer", value=False)
        send_col, clear_col = st.columns([1, 0.4])
        with send_col:
            if st.button("Send", type="primary"):
                if user_input.strip():
                    with st.spinner("Thinking..."):
                        reply = call_groq_chat(user_input, st.session_state.history, stream=stream_answer)
                    st.session_state.history.append(("user", user_input))
                    st.session_state.history.append(("assistant", reply))
                else:
                    st.warning("Please enter a question!")
        with clear_col:
            if st.button("Clear"):
                st.session_state.history = []
        st.markdown("### 📝 Conversation")
        for role, msg in st.session_state.history:
            if role == "user":
                st.markdown(f"**🧑‍💻 You:** {msg}")
            else:
                st.markdown(f"**🤖 EV-Genie:** {msg}")
    with tab2:
        st.header("🔋 Predict Real EV Range (Using ML Model)")
        if ml_model is None:
            st.error("❌ ML Model not found. Train it via `train_model.py` or place `ev_range_model.pkl` in a data folder.")
        else:
            st.success("✅ ML Model Loaded")
            battery_percent = st.slider("Battery %", 0, 100, 70)
            battery_capacity = st.number_input("Battery Capacity (kWh)", 10.0, 120.0, 40.0)
            speed = st.number_input("Speed (km/h)", 0, 200, 60)
            temp = st.number_input("Temperature (°C)", -20, 60, 25)
            if st.button("Predict Range"):
                try:
                    base_row = {
                        'Vehicle_ID': 1,
                        'Speed_kmh': speed,
                        'Acceleration_ms2': 0,
                        'Battery_State_%': battery_percent,
                        'Battery_Voltage_V': 350,
                        'Battery_Temperature_C': temp,
                        'Driving_Mode': "normal",
                        'Road_Type': "city",
                        'Traffic_Condition': "moderate",
                        'Slope_%': 0,
                        'Weather_Condition': "clear",
                        'Temperature_C': temp,
                        'Humidity_%': 50,
                        'Wind_Speed_ms': 1,
                        'Tire_Pressure_psi': 32,
                        'Vehicle_Weight_kg': 1500,
                        'Distance_Travelled_km': 1,
                        'Energy_per_km': np.nan,
                        'Battery_Drop_Rate': np.nan,
                        'Temp_Impact': abs(temp - temp),
                        'hour': 12,
                        'dayofweek': 3,
                        'month': 6,
                    }
                    df = pd.DataFrame([base_row])
                    expected_features = [
                        'Vehicle_ID','Speed_kmh','Acceleration_ms2','Battery_State_%','Battery_Voltage_V','Battery_Temperature_C',
                        'Driving_Mode','Road_Type','Traffic_Condition','Slope_%','Weather_Condition','Temperature_C','Humidity_%','Wind_Speed_ms',
                        'Tire_Pressure_psi','Vehicle_Weight_kg','Distance_Travelled_km','Battery_Drop_Rate','Temp_Impact','hour','dayofweek','month'
                    ]
                    for col in expected_features:
                        if col not in df.columns:
                            df[col] = np.nan
                    df = df[expected_features]
                    # Convert categorical-like columns
                    categorical_cols = ['Driving_Mode','Road_Type','Traffic_Condition','Weather_Condition']
                    for c in categorical_cols:
                        if c in df.columns and df[c].dtype == object:
                            df[c] = df[c].astype('category')
                    kwh_per_km = float(ml_model.predict(df)[0])
                    if not np.isfinite(kwh_per_km) or kwh_per_km <= 0:
                        st.error("⚠️ Model returned an invalid consumption value.")
                    else:
                        available_kwh = battery_capacity * (battery_percent / 100.0)
                        estimated_range = available_kwh / kwh_per_km
                        st.success(f"🚗 Estimated Range: **{estimated_range:.1f} km**")
                        st.info(f"Energy Use: **{kwh_per_km:.3f} kWh/km**")
                        _log_row("ui_prediction", {
                            'battery_percent': battery_percent,
                            'battery_capacity': battery_capacity,
                            'speed_kmh': speed,
                            'temp_c': temp,
                            'kwh_per_km': kwh_per_km,
                            'estimated_range_km': estimated_range
                        })
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
            # Scenario analysis expander
            # Feature importance (basic)
            with st.expander("Feature Importance"):
                if hasattr(ml_model, 'feature_importances_') and model_meta and 'features' in model_meta:
                    fi = ml_model.feature_importances_
                    feats = model_meta['features']
                    imp_df = pd.DataFrame({'feature': feats, 'importance': fi}).sort_values('importance', ascending=False)
                    st.bar_chart(imp_df.set_index('feature'))
                    st.dataframe(imp_df)
                    _log_row("feature_importance_view", {'top_feature': imp_df.iloc[0]['feature'] if not imp_df.empty else ''})
                else:
                    st.info("Importance not available.")

            with st.expander("Scenario Analysis (Speed & Temperature Sweeps)"):
                sweep_speeds = st.text_input("Speeds (km/h comma-separated)", value="40,60,80,100")
                sweep_temps = st.text_input("Temps (°C comma-separated)", value="0,10,20,30")
                if st.button("Run Scenarios"):
                    try:
                        speeds_list = [float(s.strip()) for s in sweep_speeds.split(',') if s.strip()]
                        temps_list = [float(t.strip()) for t in sweep_temps.split(',') if t.strip()]
                        rows = []
                        for sp in speeds_list:
                            for tp in temps_list:
                                pred = predict_range_with_model({
                                    'battery_percent': battery_percent,
                                    'battery_capacity': battery_capacity,
                                    'speed': sp,
                                    'temp': tp
                                })
                                if pred:
                                    rows.append({
                                        'speed_kmh': sp,
                                        'temp_c': tp,
                                        'kwh_per_km': pred['kwh_per_km'],
                                        'range_km': pred['estimated_range_km']
                                    })
                        if rows:
                            scenario_df = pd.DataFrame(rows)
                            st.dataframe(scenario_df.sort_values(['speed_kmh','temp_c']))
                            csv_data = scenario_df.to_csv(index=False)
                            st.download_button("Download CSV", data=csv_data, file_name="scenario_analysis.csv", mime="text/csv")
                            _log_row("scenario_run", {'rows': len(rows)})
                        else:
                            st.warning("No valid scenarios computed.")
                    except Exception as e:
                        st.error(f"Scenario error: {e}")

            # Simple route planner (aggregate distance check)
            with st.expander("Route Planner"):
                trip_distance = st.number_input("Total Trip Distance (km)", min_value=1.0, max_value=2000.0, value=150.0)
                segment_list_str = st.text_input("Optional segment distances (km, comma-separated)", value="")
                # Ensure consistent numeric types (all floats) to avoid StreamlitMixedNumericTypesError
                route_speed = st.number_input("Assumed Avg Speed (km/h)", min_value=10.0, max_value=140.0, value=float(speed))
                route_temp = st.number_input("Assumed Avg Temp (°C)", min_value=-30.0, max_value=50.0, value=float(temp))
                if st.button("Evaluate Route"):
                    try:
                        pred = predict_range_with_model({
                            'battery_percent': battery_percent,
                            'battery_capacity': battery_capacity,
                            'speed': route_speed,
                            'temp': route_temp
                        })
                        if not pred:
                            st.error("Could not compute consumption for route parameters.")
                        else:
                            est_range = pred['estimated_range_km']
                            needed_charges = 0
                            remaining = trip_distance
                            while remaining > est_range:
                                needed_charges += 1
                                remaining -= est_range
                            st.success(f"Predicted single-charge range: {est_range:.1f} km. Trip requires {needed_charges} charging stop(s).")
                            if segment_list_str.strip():
                                segs = [float(x.strip()) for x in segment_list_str.split(',') if x.strip()]
                                seg_rows = []
                                seg_cumulative = 0.0
                                charge_points = []
                                for i, dist in enumerate(segs, start=1):
                                    seg_cumulative += dist
                                    must_charge = seg_cumulative > est_range and (seg_cumulative - dist) <= est_range
                                    if must_charge:
                                        charge_points.append(i-1)  # before this segment
                                        seg_cumulative = dist
                                    seg_rows.append({'segment': i, 'segment_km': dist, 'cumulative_km': seg_cumulative})
                                st.dataframe(pd.DataFrame(seg_rows))
                                if charge_points:
                                    st.info(f"Charge after segment(s): {', '.join(map(str, charge_points))}")
                                _log_row("route_eval", {'trip_km': trip_distance, 'stops': needed_charges})
                    except Exception as e:
                        st.error(f"Route error: {e}")

if __name__ == "__main__":
    run_app()
