import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
import os
from PIL import Image
import threading
import queue
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .gesture-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<div class="main-header"><h1>🤟 Advanced Hand Gesture Recognition</h1><p>Real-time AI-powered gesture detection with MediaPipe & Machine Learning</p></div>', unsafe_allow_html=True)

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None

# MediaPipe setup
@st.cache_resource
def setup_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        model_complexity=0
    )
    return mp_hands, mp_drawing, hands

mp_hands, mp_drawing, hands = setup_mediapipe()

# Model functions
def create_sample_data():
    """Create sample gesture data for demonstration"""
    np.random.seed(42)
    gestures = ["thumbs_up", "peace", "open_palm", "fist", "ok_sign"]
    data = []
    labels = []
    
    for gesture in gestures:
        for _ in range(50):  # 50 samples per gesture
            # Generate realistic hand landmark data (21 landmarks × 3 coordinates = 63 features)
            landmarks = np.random.random(63) * 0.8 + 0.1  # Values between 0.1 and 0.9
            data.append(landmarks)
            labels.append(gesture)
    
    df = pd.DataFrame(data)
    df['label'] = labels
    df.to_csv('gesture_data.csv', index=False)
    return True

def train_gesture_model():
    """Train a gesture recognition model"""
    if not os.path.exists('gesture_data.csv'):
        st.error("No gesture data found. Creating sample data...")
        create_sample_data()
    
    try:
        # Load data
        df = pd.read_csv('gesture_data.csv')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=f_classif, k=min(45, X.shape[1]))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        accuracy = pipeline.score(X_test, y_test)
        
        # Save model
        joblib.dump(pipeline, 'gesture_model.pkl')
        
        return True, accuracy
    except Exception as e:
        return False, str(e)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        if os.path.exists('gesture_model.pkl'):
            model = joblib.load('gesture_model.pkl')
            return model, True
        else:
            return None, False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False

def extract_landmarks(hand_landmarks):
    """Extract landmark coordinates from MediaPipe hand landmarks"""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks).reshape(1, -1)

def process_frame(frame, model):
    """Process a single frame for gesture recognition"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    gesture_label = "No hand detected"
    confidence = 0.0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Extract features
            features = extract_landmarks(hand_landmarks)
            
            if features.shape[1] == 63:  # Ensure correct number of features
                try:
                    prediction = model.predict(features)[0]
                    if hasattr(model, 'predict_proba'):
                        confidence = np.max(model.predict_proba(features))
                        gesture_label = f"{prediction} ({int(confidence*100)}%)"
                    else:
                        gesture_label = str(prediction)
                except Exception as e:
                    gesture_label = f"Prediction error: {str(e)[:20]}"
    
    return frame, gesture_label, confidence

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    st.subheader("📊 Model Management")
    
    if st.button("🔄 Create Sample Data"):
        with st.spinner("Creating sample gesture data..."):
            if create_sample_data():
                st.success("✅ Sample data created!")
                st.info("📄 Data saved as 'gesture_data.csv'")
    
    if st.button("🧠 Train Model"):
        with st.spinner("Training gesture recognition model..."):
            success, result = train_gesture_model()
            if success:
                st.success(f"✅ Model trained! Accuracy: {result:.2%}")
                st.session_state.model_loaded = False  # Force reload
            else:
                st.error(f"❌ Training failed: {result}")
    
    # Model status
    if not st.session_state.model_loaded:
        model, loaded = load_model()
        st.session_state.model = model
        st.session_state.model_loaded = loaded
    
    if st.session_state.model_loaded:
        st.success("✅ Model loaded successfully!")
    else:
        st.warning("⚠️ No model found. Please train a model first.")
    
    st.subheader("🎭 Supported Gestures")
    gestures_info = {
        "👍": "Thumbs Up",
        "✌️": "Peace Sign", 
        "✋": "Open Palm",
        "👊": "Fist",
        "👌": "OK Sign"
    }
    
    for emoji, name in gestures_info.items():
        st.markdown(f'<div class="gesture-box">{emoji} <strong>{name}</strong></div>', unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📹 Live Gesture Recognition")
    
    # Camera controls
    camera_col1, camera_col2 = st.columns(2)
    
    with camera_col1:
        start_camera = st.button("📷 Start Camera", disabled=st.session_state.camera_active)
    
    with camera_col2:
        stop_camera = st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_active)
    
    # Video frame placeholder
    frame_placeholder = st.empty()
    
    # Gesture result placeholder
    result_placeholder = st.empty()

with col2:
    st.header("📈 Live Statistics")
    
    # Metrics placeholders
    confidence_metric = st.empty()
    fps_metric = st.empty()
    gesture_history = st.empty()
    
    # Instructions
    st.header("💡 Instructions")
    st.markdown("""
    1. **Train Model**: Click 'Create Sample Data' then 'Train Model' in sidebar
    2. **Start Camera**: Click 'Start Camera' button
    3. **Show Gesture**: Make gestures in front of camera
    4. **View Results**: See real-time predictions and confidence scores
    
    **Tips for best results:**
    - Ensure good lighting
    - Keep hand clearly visible
    - Make distinct gestures
    - Stay within camera frame
    """)

# Camera logic
if start_camera and st.session_state.model_loaded:
    st.session_state.camera_active = True

if stop_camera:
    st.session_state.camera_active = False

# Main camera loop
if st.session_state.camera_active and st.session_state.model_loaded:
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        # Performance tracking
        fps_counter = 0
        start_time = time.time()
        gesture_history_list = []
        
        if not cap.isOpened():
            st.error("❌ Cannot access camera. Please check camera permissions.")
            st.session_state.camera_active = False
        else:
            placeholder_info = st.info("📷 Camera is active. Show your hand gestures!")
            
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Mirror the frame
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame, gesture_label, confidence = process_frame(frame, st.session_state.model)
                
                # Add text overlay
                cv2.putText(processed_frame, gesture_label, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"FPS: {fps_counter}", (10, processed_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Update display
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Update metrics
                with confidence_metric:
                    st.metric("🎯 Confidence", f"{confidence:.1%}")
                
                with fps_metric:
                    current_time = time.time()
                    if current_time - start_time >= 1.0:
                        fps_counter = int(fps_counter / (current_time - start_time))
                        st.metric("⚡ FPS", fps_counter)
                        fps_counter = 0
                        start_time = current_time
                    else:
                        fps_counter += 1
                
                # Update gesture history
                if "No hand" not in gesture_label:
                    gesture_history_list.append(gesture_label)
                    if len(gesture_history_list) > 10:
                        gesture_history_list.pop(0)
                
                with gesture_history:
                    st.subheader("📋 Recent Gestures")
                    for i, gesture in enumerate(reversed(gesture_history_list[-5:])):
                        st.text(f"{5-i}. {gesture}")
                
                # Display current result
                with result_placeholder:
                    if "No hand" not in gesture_label:
                        st.success(f"🎯 **Detected**: {gesture_label}")
                    else:
                        st.info("👋 Show your hand to start recognition")
                
                # Small delay to prevent overwhelming the display
                time.sleep(0.033)  # ~30 FPS
                
                # Check if we should stop (re-run detection)
                if not st.session_state.camera_active:
                    break
            
            cap.release()
            placeholder_info.success("📷 Camera stopped successfully")
            
    except Exception as e:
        st.error(f"❌ Camera error: {e}")
        st.session_state.camera_active = False

elif st.session_state.camera_active and not st.session_state.model_loaded:
    st.error("❌ No model loaded. Please train a model first using the sidebar.")
    st.session_state.camera_active = False

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🤟 Hand Gesture Recognition App | Built with Streamlit, OpenCV, MediaPipe & scikit-learn</p>
    <p>For best performance, ensure good lighting and clear hand visibility</p>
</div>
""", unsafe_allow_html=True)
