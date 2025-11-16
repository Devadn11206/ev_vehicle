"""Lightweight smoke test for chatbot ML integration without Streamlit UI.
Run: python EV-Range-Prediction-AI/src/chatbot_smoke_test.py
Optionally set GROQ_API_KEY beforehand for live LLM call.
"""
import os
import sys
import importlib
from pathlib import Path

# Adjust path so we can import chatbot module
repo_root = Path(__file__).resolve().parent
sys.path.append(str(repo_root))

cb = importlib.import_module("chatbot")

print("== Smoke Test: Chatbot ML Integration ==")
print(f"Groq key present: {'YES' if os.getenv('GROQ_API_KEY') else 'NO'}")
print(f"Model loaded: {'YES' if cb.ml_model is not None else 'NO'}")

queries = [
    "Estimate range 70% 40kWh at 60km/h 25C",
    "How far can I go with 55% battery on a 50kWh pack at 80km/h 10C?",
    "Explain how temperature affects EV range",
]

for q in queries:
    print("\n--- Query ---")
    print(q)
    # We call internal helpers to show parsed params and predicted range
    params = cb.extract_parameters(q)
    prediction = cb.predict_range_with_model(params)
    print("Parsed params:", params)
    if prediction:
        print(f"Prediction: {prediction['estimated_range_km']:.1f} km (consumption {prediction['kwh_per_km']:.3f} kWh/km)")
    else:
        print("Prediction: <none>")
    # Call LLM if key exists else skip
    if os.getenv('GROQ_API_KEY'):
        resp = cb.call_groq_chat(q, history=None, stream=False)
        print("LLM reply (truncated):", resp[:300].replace('\n',' '))
    else:
        print("LLM reply skipped (no API key set).")

print("\nSmoke test complete.")
