import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
st.markdown('<div class="main-header"><h1>🤟 Hand Gesture Recognition</h1><p>AI-powered gesture detection with MediaPipe & Machine Learning</p></div>', unsafe_allow_html=True)

# Deployment notice
st.info("📱 **Streamlit Cloud Deployment**: Live camera is not available in cloud deployment. Use the image upload feature for gesture recognition!")

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
    try:
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
    except Exception as e:
        st.error(f"Error setting up MediaPipe: {e}")
        return None, None, None

try:
    mp_hands, mp_drawing, hands = setup_mediapipe()
    if mp_hands is None:
        st.error("Failed to initialize MediaPipe. Please refresh the page.")
        st.stop()
except Exception as e:
    st.error(f"MediaPipe initialization failed: {e}")
    st.stop()

# Model functions
@st.cache_data
def create_sample_data():
    """Create sample gesture data for demonstration"""
    try:
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
    except Exception as e:
        st.error(f"Error creating sample data: {e}")
        return False

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

def test_camera():
    """Test camera access and return status"""
    try:
        # Simplified camera test for deployment
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return {
                    'success': True, 
                    'index': 0,
                    'width': width,
                    'height': height
                }
            cap.release()
        return {'success': False, 'error': 'Camera not accessible in deployment environment'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def process_frame(frame, model):
    """Process a single frame for gesture recognition"""
    try:
        if hands is None:
            return frame, "MediaPipe not initialized", 0.0
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        gesture_label = "No hand detected"
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                if mp_drawing and mp_hands:
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
                        gesture_label = f"Prediction error"
        
        return frame, gesture_label, confidence
    except Exception as e:
        return frame, f"Processing error", 0.0

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
    
    if st.button("📹 Test Camera"):
        with st.spinner("Testing camera access..."):
            camera_test_result = test_camera()
            if camera_test_result['success']:
                st.success(f"✅ Camera working! Found at index {camera_test_result['index']}")
                st.info(f"Resolution: {camera_test_result['width']}x{camera_test_result['height']}")
            else:
                st.error(f"❌ Camera test failed: {camera_test_result['error']}")
                st.info("💡 Try running the app locally for better camera support")
    
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

# Main camera loop - Simplified for deployment
if st.session_state.camera_active and st.session_state.model_loaded:
    st.warning("� **Note**: Live camera is not available in Streamlit Cloud deployment.")
    st.info("💡 **Solution**: Use the image upload feature below for gesture recognition!")
    st.session_state.camera_active = False
    
    # Always show image upload alternative
    st.subheader("📸 Upload Image for Gesture Recognition")
    uploaded_file = st.file_uploader("Choose an image with hand gesture", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        try:
            # Read the image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            if len(image_np.shape) == 3:
                # Convert RGB to BGR for OpenCV
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # Process the image
                processed_frame, gesture_label, confidence = process_frame(image_bgr, st.session_state.model)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Original Image", use_column_width=True)
                
                with col2:
                    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    st.image(processed_rgb, caption="Processed Image with Hand Landmarks", use_column_width=True)
                
                # Show results
                if "No hand" not in gesture_label and "error" not in gesture_label.lower():
                    st.success(f"🎯 **Detected Gesture**: {gesture_label}")
                    st.metric("🎯 Confidence", f"{confidence:.1%}")
                else:
                    st.info("👋 No hand gesture detected in the image")
                    
        except Exception as e:
            st.error(f"Error processing image: {e}")

elif st.session_state.camera_active and not st.session_state.model_loaded:
    st.error("❌ No model loaded. Please train a model first using the sidebar.")
    st.session_state.camera_active = False

# Always show image upload section for deployment
if not st.session_state.camera_active:
    st.header("📸 Image-Based Gesture Recognition")
    st.info("💡 Upload an image with hand gestures to test the AI model!")
    
    uploaded_file = st.file_uploader("Choose an image with hand gesture", type=['jpg', 'jpeg', 'png'], key="main_uploader")
    
    if uploaded_file is not None and st.session_state.model_loaded:
        try:
            # Read the image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            if len(image_np.shape) == 3:
                # Convert RGB to BGR for OpenCV
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # Process the image
                processed_frame, gesture_label, confidence = process_frame(image_bgr, st.session_state.model)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="📷 Original Image", use_column_width=True)
                
                with col2:
                    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    st.image(processed_rgb, caption="🎯 AI Analysis with Hand Landmarks", use_column_width=True)
                
                # Show results with better formatting
                st.markdown("---")
                if "No hand" not in gesture_label and "error" not in gesture_label.lower():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"🎯 **Detected Gesture**: {gesture_label}")
                    with col2:
                        st.metric("📊 Confidence Score", f"{confidence:.1%}")
                else:
                    st.info("👋 No hand gesture detected in the image. Try with a clearer hand gesture image.")
                    
        except Exception as e:
            st.error(f"Error processing image: {e}")
    elif uploaded_file is not None and not st.session_state.model_loaded:
        st.warning("⚠️ Please train a model first using the sidebar controls!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🤟 Hand Gesture Recognition App | Built with Streamlit, OpenCV, MediaPipe & scikit-learn</p>
    <p>For best performance, ensure good lighting and clear hand visibility</p>
</div>
""", unsafe_allow_html=True)
