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
import plotly.express as px
import plotly.graph_objects as go
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
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .gesture-box {
        background: linear-gradient(135deg, #e8f4fd 0%, #f0f8ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .camera-status {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    .status-active {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-inactive {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'gesture_history' not in st.session_state:
    st.session_state.gesture_history = []
if 'fps_history' not in st.session_state:
    st.session_state.fps_history = []

# Title and header
st.markdown('''
<div class="main-header">
    <h1>🤟 Advanced Hand Gesture Recognition</h1>
    <p>Real-time AI-powered gesture detection with MediaPipe & Machine Learning</p>
    <p><em>Smooth • Responsive • Accurate</em></p>
</div>
''', unsafe_allow_html=True)

# MediaPipe setup
@st.cache_resource
def setup_mediapipe():
    """Initialize MediaPipe with optimal settings"""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        model_complexity=0  # Fastest model
    )
    return mp_hands, mp_drawing, hands

mp_hands, mp_drawing, hands = setup_mediapipe()

# Helper functions
def create_sample_data():
    """Create sample gesture data for demonstration"""
    np.random.seed(42)
    gestures = ["thumbs_up", "peace", "open_palm", "fist", "ok_sign"]
    all_data = []
    all_labels = []
    
    for gesture in gestures:
        for _ in range(100):  # 100 samples per gesture
            # Generate realistic hand landmark data (63 features)
            base_landmarks = np.random.rand(63) * 0.8 + 0.1  # Values between 0.1 and 0.9
            
            # Add gesture-specific patterns
            if gesture == "thumbs_up":
                base_landmarks[4*3:5*3] += 0.1  # Thumb up
            elif gesture == "peace":
                base_landmarks[8*3:9*3] += 0.1  # Index up
                base_landmarks[12*3:13*3] += 0.1  # Middle up
            elif gesture == "fist":
                base_landmarks *= 0.8  # Fingers down
            
            # Add some noise
            noise = np.random.normal(0, 0.02, 63)
            features = np.clip(base_landmarks + noise, 0, 1)
            
            all_data.append(features)
            all_labels.append(gesture)
    
    df = pd.DataFrame(all_data)
    df['label'] = all_labels
    return df

def train_model_from_data(df):
    """Train a gesture recognition model from data"""
    if df is None or df.empty:
        return None
        
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif, k=50)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    accuracy = pipeline.score(X_test, y_test)
    
    return pipeline, accuracy

def load_or_create_model():
    """Load existing model or create new one"""
    model_path = "gesture_model.pkl"
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.session_state.model = model
            st.session_state.model_loaded = True
            return model, "loaded"
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    # Check for data file
    if os.path.exists("gesture_data.csv"):
        try:
            df = pd.read_csv("gesture_data.csv")
            if len(df) > 0:
                model, accuracy = train_model_from_data(df)
                if model:
                    joblib.dump(model, model_path)
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    return model, f"trained (accuracy: {accuracy:.2%})"
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    # Create sample model
    try:
        df = create_sample_data()
        model, accuracy = train_model_from_data(df)
        if model:
            joblib.dump(model, model_path)
            st.session_state.model = model
            st.session_state.model_loaded = True
            return model, f"demo model created (accuracy: {accuracy:.2%})"
    except Exception as e:
        st.error(f"Error creating demo model: {e}")
    
    return None, "failed"

def extract_landmarks(hand_landmarks):
    """Extract landmark coordinates from MediaPipe hand landmarks"""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks)

def predict_gesture(landmarks, model):
    """Predict gesture from landmarks"""
    if model is None or len(landmarks) != 63:
        return "No hand", 0.0
    
    try:
        features = landmarks.reshape(1, -1)
        prediction = model.predict(features)[0]
        
        # Get confidence if available
        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(features))
        else:
            confidence = 0.8  # Default confidence
            
        return prediction, confidence
    except Exception as e:
        return "Error", 0.0

def process_frame(frame, model, hands):
    """Process frame for gesture recognition"""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(rgb_frame)
    
    gesture_label = "No hand detected"
    confidence = 0.0
    
    # Draw landmarks and predict
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Extract features and predict
            landmarks = extract_landmarks(hand_landmarks)
            gesture_label, confidence = predict_gesture(landmarks, model)
            
            # Only use first hand
            break
    
    return frame, gesture_label, confidence

# Sidebar
st.sidebar.title("🎛️ Control Panel")

# Model Management
st.sidebar.subheader("🧠 Model Management")

if st.sidebar.button("🔥 Load/Create Model"):
    with st.spinner("Loading or creating model..."):
        model, status = load_or_create_model()
        if model:
            st.sidebar.success(f"✅ Model {status}")
        else:
            st.sidebar.error("❌ Failed to load/create model")

# Model status
if st.session_state.model_loaded:
    st.sidebar.success("✅ Model Ready")
else:
    st.sidebar.warning("⚠️ No Model Loaded")

# Camera controls
st.sidebar.subheader("📹 Camera Controls")

camera_col1, camera_col2 = st.sidebar.columns(2)

with camera_col1:
    if st.button("▶️ Start Camera"):
        if st.session_state.model_loaded:
            st.session_state.camera_active = True
            st.rerun()
        else:
            st.error("Load model first!")

with camera_col2:
    if st.button("⏹️ Stop Camera"):
        st.session_state.camera_active = False
        st.rerun()

# Supported gestures
st.sidebar.subheader("🎭 Supported Gestures")
gestures = ["👍 Thumbs Up", "✌️ Peace", "✋ Open Palm", "✊ Fist", "👌 OK Sign"]
for gesture in gestures:
    st.sidebar.markdown(f"• {gesture}")

# Performance settings
st.sidebar.subheader("⚙️ Performance")
fps_limit = st.sidebar.slider("FPS Limit", 10, 30, 20, help="Higher FPS = smoother but more CPU intensive")
resolution = st.sidebar.selectbox("Resolution", ["640x480", "800x600", "1280x720"], index=0)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    
    # Camera status
    if st.session_state.camera_active:
        st.markdown('<div class="camera-status status-active">🟢 Camera Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="camera-status status-inactive">🔴 Camera Inactive</div>', unsafe_allow_html=True)
    
    # Video display placeholder
    frame_placeholder = st.empty()

with col2:
    st.subheader("📊 Real-time Metrics")
    
    # Metrics placeholders
    confidence_metric = st.empty()
    fps_metric = st.empty()
    gesture_metric = st.empty()
    
    # Gesture history
    st.subheader("📋 Recent Gestures")
    history_placeholder = st.empty()

# Camera processing
if st.session_state.camera_active and st.session_state.model_loaded:
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        # Set resolution
        width, height = map(int, resolution.split('x'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps_limit)
        
        if not cap.isOpened():
            st.error("❌ Cannot access camera. Please check camera permissions.")
            st.session_state.camera_active = False
        else:
            # Performance tracking
            fps_counter = 0
            start_time = time.time()
            frame_time = 1.0 / fps_limit
            last_frame_time = time.time()
            
            # Processing loop
            while st.session_state.camera_active:
                # Frame rate control
                current_time = time.time()
                if current_time - last_frame_time < frame_time:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Mirror the frame for better UX
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame, gesture_label, confidence = process_frame(
                    frame.copy(), st.session_state.model, hands
                )
                
                # Add text overlays
                cv2.putText(processed_frame, f"Gesture: {gesture_label}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Confidence: {confidence:.1%}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"FPS: {fps_counter}", (10, processed_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Update display
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Update metrics
                with confidence_metric:
                    st.metric("🎯 Confidence", f"{confidence:.1%}")
                
                with gesture_metric:
                    if "No hand" not in gesture_label and "Error" not in gesture_label:
                        st.success(f"🎭 {gesture_label}")
                    else:
                        st.info("👋 Show your hand")
                
                # FPS calculation
                fps_counter += 1
                if current_time - start_time >= 1.0:
                    actual_fps = fps_counter / (current_time - start_time)
                    with fps_metric:
                        st.metric("⚡ FPS", f"{actual_fps:.1f}")
                    
                    # Store FPS history
                    st.session_state.fps_history.append(actual_fps)
                    if len(st.session_state.fps_history) > 50:
                        st.session_state.fps_history.pop(0)
                    
                    fps_counter = 0
                    start_time = current_time
                
                # Update gesture history
                if "No hand" not in gesture_label and "Error" not in gesture_label:
                    if not st.session_state.gesture_history or st.session_state.gesture_history[-1] != gesture_label:
                        st.session_state.gesture_history.append(gesture_label)
                        if len(st.session_state.gesture_history) > 20:
                            st.session_state.gesture_history.pop(0)
                
                # Display gesture history
                with history_placeholder:
                    if st.session_state.gesture_history:
                        recent_gestures = st.session_state.gesture_history[-5:]
                        for i, gesture in enumerate(reversed(recent_gestures)):
                            st.text(f"{len(recent_gestures)-i}. {gesture}")
                    else:
                        st.text("No gestures detected yet")
                
                last_frame_time = current_time
                
                # Small delay to prevent overwhelming
                time.sleep(0.001)
                
                # Check if we should stop
                if not st.session_state.camera_active:
                    break
            
            cap.release()
            frame_placeholder.success("📷 Camera stopped successfully")
            
    except Exception as e:
        st.error(f"❌ Camera error: {e}")
        st.session_state.camera_active = False

elif st.session_state.camera_active and not st.session_state.model_loaded:
    st.error("❌ No model loaded. Please load a model first using the sidebar.")
    st.session_state.camera_active = False

# Performance visualization
if st.session_state.fps_history:
    st.subheader("📈 Performance Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # FPS chart
        fig = px.line(
            x=list(range(len(st.session_state.fps_history))),
            y=st.session_state.fps_history,
            title="FPS Over Time",
            labels={'x': 'Time (seconds)', 'y': 'FPS'}
        )
        fig.update_traces(line_color='#667eea')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gesture distribution
        if st.session_state.gesture_history:
            gesture_counts = pd.Series(st.session_state.gesture_history).value_counts()
            fig = px.pie(
                values=gesture_counts.values,
                names=gesture_counts.index,
                title="Gesture Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <h4>🤟 Hand Gesture Recognition App</h4>
    <p>Built with ❤️ using Streamlit, OpenCV, MediaPipe & scikit-learn</p>
    <p><em>For best performance, ensure good lighting and clear hand visibility</em></p>
    <p><strong>Tips:</strong> Keep your hand steady, use contrasting background, maintain consistent distance</p>
</div>
""", unsafe_allow_html=True)
