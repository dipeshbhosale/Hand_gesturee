import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os
import pandas as pd

# Import main.py functions
import sys
import importlib.util

MAIN_PATH = os.path.join(os.path.dirname(__file__), "main.py")
spec = importlib.util.spec_from_file_location("main", MAIN_PATH)
main = importlib.util.module_from_spec(spec)
sys.modules["main"] = main
spec.loader.exec_module(main)

st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")
st.title("🤟 Real-Time Hand Gesture Recognition (Streamlit)")

st.markdown("""
- This Streamlit app uses your webcam to recognize hand gestures in real time.
- Make sure you have a trained model (`gesture_model.pkl`) in the same directory.
- For best results, use on your local machine with a webcam.
""")

if st.button('Collect Gesture Data (Integrated)'):
    with st.spinner("Collecting gesture data using your webcam..."):
        result = main.collect_gesture_data_integrated()
        if result:
            st.success("Gesture data collected and saved to gesture_data.csv!")
        else:
            st.error("Gesture data collection failed or was cancelled.")

if st.button('Train Model (Advanced)'):
    with st.spinner("Training model with advanced pipeline..."):
        result = main.train_model_advanced()
        if result:
            st.success("Model trained and saved as gesture_model.pkl!")
        else:
            st.error("Model training failed.")

run = st.checkbox('Start Camera for Real-Time Prediction')
FRAME_WINDOW = st.image([])
label_placeholder = st.empty()

@st.cache_resource
def load_model():
    return joblib.load("gesture_model.pkl")

model = None
try:
    model = load_model()
except Exception as e:
    st.warning(f"Model not loaded: {e}")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

if run and model is not None:
    cap = cv2.VideoCapture(0)
    st.info("Press 'Stop Camera' to release webcam.")
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        gesture_label = "No hand"
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                if len(landmarks) == 63:
                    features_np = np.array(landmarks).reshape(1, -1)
                    pred = model.predict(features_np)[0]
                    gesture_label = str(pred)
        cv2.putText(frame, gesture_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        FRAME_WINDOW.image(frame, channels="BGR")
        label_placeholder.markdown(f"### Detected Gesture: `{gesture_label}`")
        time.sleep(0.05)
    cap.release()
    st.success("Webcam released.")
