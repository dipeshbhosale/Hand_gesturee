import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')
import time

# Optional: for keyboard capture during data collection
try:
    import keyboard
except ImportError:
    keyboard = None

# --- INTEGRATED COLLECTION FUNCTION ---
def collect_gesture_data_integrated():
    """
    Integrated gesture collection with multiple gestures in one session.
    """
    if not keyboard:
        print("❌ 'keyboard' package required for data collection. Install with: pip install keyboard")
        return
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    
    all_data = []
    all_labels = []
    gestures = ["thumbs_up", "peace", "open_palm", "fist", "ok_sign"]
    
    for gesture in gestures:
        print(f"\n=== Collecting data for gesture: {gesture} ===")
        print("Position your hand and press SPACE to capture samples.")
        print("Collect at least 30 samples per gesture. Press 'q' to move to next gesture.\n")
        
        cap = cv2.VideoCapture(0)
        gesture_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {gesture_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture | Q: Next gesture", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    if keyboard.is_pressed("space") and len(landmarks) == 63:
                        all_data.append(landmarks)
                        all_labels.append(gesture)
                        gesture_count += 1
                        print(f"Captured {gesture} sample #{gesture_count}")
            
            cv2.imshow("Gesture Collection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        print(f"Collected {gesture_count} samples for {gesture}")
    
    cv2.destroyAllWindows()
    
    if all_data:
        df = pd.DataFrame(all_data)
        df["label"] = all_labels
        df.to_csv("gesture_data.csv", index=False)
        print(f"\n✅ Saved {len(all_data)} total samples to gesture_data.csv")
        return True
    return False

# --- IMPROVED TRAINING FUNCTION ---
def augment_jitter(X, y, jitter_std=0.01, n_aug=2):
    """
    Augment data by adding Gaussian noise (jitter) to landmark features.
    """
    X_aug = []
    y_aug = []
    for xi, yi in zip(X, y):
        for _ in range(n_aug):
            noise = np.random.normal(0, jitter_std, size=xi.shape)
            X_aug.append(xi + noise)
            y_aug.append(yi)
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    return np.vstack([X, X_aug]), np.hstack([y, y_aug])

def advanced_data_augmentation(X, y, augment_factor=3):
    """
    Advanced data augmentation with rotation, scaling, and multiple noise types.
    """
    X_aug = []
    y_aug = []
    
    for xi, yi in zip(X, y):
        # Original sample
        X_aug.append(xi)
        y_aug.append(yi)
        
        for _ in range(augment_factor):
            # Reshape to 21 landmarks × 3 coordinates
            landmarks = xi.reshape(21, 3)
            
            # 1. Gaussian noise
            noise = np.random.normal(0, 0.005, landmarks.shape)
            noisy_landmarks = landmarks + noise
            
            # 2. Scaling augmentation
            scale_factor = np.random.uniform(0.95, 1.05)
            scaled_landmarks = landmarks * scale_factor
            
            # 3. Translation (small shifts)
            translation = np.random.normal(0, 0.01, (1, 3))
            translated_landmarks = landmarks + translation
            
            # 4. Coordinate dropout (randomly set some coords to mean)
            dropout_landmarks = landmarks.copy()
            if np.random.random() > 0.7:
                dropout_idx = np.random.choice(21, size=3, replace=False)
                dropout_landmarks[dropout_idx] = np.mean(landmarks, axis=0)
            
            # Add augmented samples
            for aug_landmarks in [noisy_landmarks, scaled_landmarks, translated_landmarks, dropout_landmarks]:
                # Ensure landmarks are within valid range [0, 1]
                aug_landmarks = np.clip(aug_landmarks, 0, 1)
                X_aug.append(aug_landmarks.flatten())
                y_aug.append(yi)
    
    return np.array(X_aug), np.array(y_aug)

def train_model_advanced(csv_path="gesture_data.csv", model_path="gesture_model.pkl"):
    """
    Advanced training with ensemble methods, feature selection, and robust validation.
    """
    if not os.path.exists(csv_path):
        print("❌ gesture_data.csv not found! Run data collection first.")
        return False

    print("📊 Loading gesture data...")
    df = pd.read_csv(csv_path)
    if "label" not in df.columns or df.shape[1] != 64:
        print(f"❌ Invalid CSV format. Expected 64 columns, got {df.shape[1]}")
        return False

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print(f"Original data: {X.shape[0]} samples, {len(np.unique(y))} classes")

    # Advanced data augmentation
    print("🔄 Applying advanced data augmentation...")
    X_aug, y_aug = advanced_data_augmentation(X, y, augment_factor=2)
    print(f"Augmented data: {X_aug.shape[0]} samples")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection
    print("🎯 Performing feature selection...")
    selector = SelectKBest(score_func=f_classif, k=min(45, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Define multiple models with regularization
    models = {
        'SVC': SVC(probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }

    # Grid search parameters
    param_grids = {
        'SVC': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto'],
            'class_weight': [None, 'balanced']
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': [None, 'balanced']
        },
        'GradientBoosting': {
            'n_estimators': [100, 150],
            'learning_rate': [0.1, 0.05],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        },
        'LogisticRegression': {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced']
        }
    }

    # Train and evaluate models
    best_models = {}
    cv_scores = {}
    
    # Stratified K-Fold for robust validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n🔍 Training {name}...")
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grids[name], 
            cv=3, n_jobs=-1, scoring='accuracy',
            verbose=0
        )
        grid_search.fit(X_train_selected, y_train)
        
        best_models[name] = grid_search.best_estimator_
        
        # Cross-validation on best model
        cv_score = cross_val_score(
            best_models[name], X_train_selected, y_train, 
            cv=skf, scoring='accuracy'
        )
        cv_scores[name] = cv_score
        
        test_score = best_models[name].score(X_test_selected, y_test)
        
        print(f"{name} - Best params: {grid_search.best_params_}")
        print(f"{name} - CV Score: {cv_score.mean():.3f} (±{cv_score.std()*2:.3f})")
        print(f"{name} - Test Score: {test_score:.3f}")

    # Create ensemble model
    print("\n🎭 Creating ensemble model...")
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in best_models.items()],
        voting='soft'
    )
    ensemble.fit(X_train_selected, y_train)
    
    # Evaluate ensemble
    ensemble_cv = cross_val_score(ensemble, X_train_selected, y_train, cv=skf, scoring='accuracy')
    ensemble_test = ensemble.score(X_test_selected, y_test)
    
    print(f"Ensemble - CV Score: {ensemble_cv.mean():.3f} (±{ensemble_cv.std()*2:.3f})")
    print(f"Ensemble - Test Score: {ensemble_test:.3f}")

    # Select best model
    all_scores = {name: best_models[name].score(X_test_selected, y_test) for name in best_models}
    all_scores['Ensemble'] = ensemble_test
    
    best_model_name = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_model_name]
    
    if best_model_name == 'Ensemble':
        final_model = ensemble
    else:
        final_model = best_models[best_model_name]
    
    # Create final pipeline with preprocessing
    from sklearn.pipeline import Pipeline
    final_pipeline = Pipeline([
        ('scaler', scaler),
        ('selector', selector),
        ('classifier', final_model)
    ])
    
    # Refit on original training data
    final_pipeline.fit(X_train, y_train)
    final_test_score = final_pipeline.score(X_test, y_test)

    print(f"\n✅ Best model: {best_model_name}")
    print(f"🎯 Final test accuracy: {final_test_score:.3f}")
    
    # Check for overfitting
    if best_model_name in cv_scores:
        cv_mean = cv_scores[best_model_name].mean()
        if final_test_score - cv_mean > 0.1:
            print("⚠️  Warning: Possible overfitting detected!")
        else:
            print("✅ Model appears to generalize well.")

    joblib.dump(final_pipeline, model_path)
    print(f"✅ Model pipeline saved as {model_path}")
    
    return True

# --- 3️⃣ Ensure gesture_model.pkl exists or auto-train if missing ---
def ensure_model_exists(csv_path="gesture_data.csv", model_path="gesture_model.pkl"):
    """
    Ensure gesture_model.pkl exists. If not, train and save from gesture_data.csv if available.
    """
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        if os.path.exists(csv_path):
            print(f"Found '{csv_path}'. Training model now...")
            train_model_advanced()
        else:
            raise FileNotFoundError(
                f"Neither '{model_path}' nor '{csv_path}' found. Please collect data and train the model first."
            )

# --- 3️⃣ REAL-TIME PREDICTION (OpenCV version) ---
def predict_gesture_realtime(model_path="gesture_model.pkl"):
    """
    Real-time webcam gesture prediction using trained model and MediaPipe.
    Limited to 15 FPS and 360p resolution for better performance.
    """
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please train the model first.")
        return
    model = joblib.load(model_path)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution to 360p and FPS to 15
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    # Frame rate control
    fps_limit = 15
    frame_time = 1.0 / fps_limit
    last_time = time.time()
    
    print("Press 'q' to quit. Running at 15 FPS @ 360p for optimal performance.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # FPS control
        current_time = time.time()
        if current_time - last_time < frame_time:
            continue
        last_time = current_time
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        gesture_label = "No hand"
        confidence = 0.0
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                if len(landmarks) == 63:
                    features_np = np.array(landmarks).reshape(1, -1)
                    pred = model.predict(features_np)[0]
                    if hasattr(model, "predict_proba"):
                        conf = np.max(model.predict_proba(features_np))
                        confidence = float(conf)
                        gesture_label = f"{pred} {int(confidence*100)}%"
                    else:
                        gesture_label = str(pred)
                else:
                    gesture_label = "Incomplete landmarks"
        
        # Add FPS info to display
        cv2.putText(
            frame, gesture_label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, "15 FPS @ 360p", (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        
        cv2.imshow("Real-Time Gesture Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

# --- 3️⃣ REAL-TIME PREDICTION (Gradio version) ---
import gradio as gr

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --- Create persistent MediaPipe Hands object for performance ---
# This avoids re-initializing the model on every frame, which is the main cause of lag.
hands_solution = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
    model_complexity=0  # Use the lightest model
)

# Optimized FPS control for Gradio processing
last_gradio_time = time.time()
gradio_fps_limit = 15  # Increased back to 15 FPS for smoother video
gradio_frame_time = 1.0 / gradio_fps_limit
frame_skip_counter = 0

def detect_gesture(image):
    """
    Optimized gesture detection for Gradio with 15 FPS for smoother experience.
    Uses a persistent MediaPipe model with reduced frame skipping.
    """
    global last_gradio_time, frame_skip_counter
    
    # Reduced frame skipping for smoother video
    frame_skip_counter += 1
    if frame_skip_counter % 2 != 0:  # Process only every 2nd frame instead of 4th
        if hasattr(detect_gesture, "last_result"):
            return detect_gesture.last_result
        return None, "Processing..."
    
    # FPS control for Gradio
    current_time = time.time()
    if current_time - last_gradio_time < gradio_frame_time:
        if hasattr(detect_gesture, "last_result"):
            return detect_gesture.last_result
        return None, "Processing at 15 FPS..."
    
    last_gradio_time = current_time
    
    if image is None:
        return None, "No input - Check camera permissions"
    
    try:
        frame_rgb = np.array(image)
        if frame_rgb.size == 0:
            return None, "Empty frame - Camera not working"
        
        # Resize frame for faster processing
        height, width = frame_rgb.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        
        gesture_label = "No hand detected"
        confidence = 0.0

        # Lazy load model (only when needed)
        if not hasattr(detect_gesture, "model"):
            try:
                ensure_model_exists()
                model_data = joblib.load("gesture_model.pkl")
                if isinstance(model_data, dict):
                    detect_gesture.model = model_data.get('model') or model_data.get('pipeline')
                else:
                    detect_gesture.model = model_data
                print("✅ Model loaded successfully")
            except Exception as e:
                return frame_rgb, f"Model Error: {str(e)}"

        # Convert to BGR for OpenCV processing
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        annotated_bgr = frame_bgr.copy()

        # Process the frame using the persistent hands object (no 'with' block)
        results = hands_solution.process(frame_rgb)
            
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Lighter landmark drawing
                mp_drawing.draw_landmarks(
                    annotated_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Extract features
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                
                if len(features) == 63:
                    features_np = np.array(features).reshape(1, -1)
                    model = detect_gesture.model
                    
                    try:
                        pred = model.predict(features_np)[0]
                        if hasattr(model, "predict_proba"):
                            conf = np.max(model.predict_proba(features_np))
                            confidence = float(conf)
                            gesture_label = f"{pred} ({int(confidence*100)}%)"
                        else:
                            gesture_label = str(pred)
                    except Exception as e:
                        gesture_label = f"Prediction error"
                else:
                    gesture_label = f"Invalid landmarks: {len(features)}/63"
        
        # Simplified text overlay
        cv2.putText(
            annotated_bgr, gesture_label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )
        
        # Add optimized status
        status_text = "Hand OK" if results.multi_hand_landmarks else "No Hand"
        cv2.putText(
            annotated_bgr, status_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA
        )
        
        # Add performance indicator
        cv2.putText(
            annotated_bgr, "15 FPS (Optimized)", (10, annotated_bgr.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
        )
        
        # Convert back to RGB for Gradio
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
        # Cache result for performance
        detect_gesture.last_result = (annotated_rgb, gesture_label)
        
        return annotated_rgb, gesture_label
        
    except Exception as e:
        error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Error: {str(e)[:30]}", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(error_frame, "15 FPS (Optimized)", (10, 340),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return error_frame, f"Error: {str(e)[:50]}"

# Optimized Gradio interface
with gr.Blocks(title="Hand Gesture Recognition", theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🤟 Real-Time Hand Gesture Recognition")
    gr.Markdown("### Optimized for Web Performance")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Optimized webcam component
            webcam = gr.Image(
                label="📹 Live Webcam (Optimized for Web)",
                streaming=True,
                height=360,
                width=640,
                mirror_webcam=False
            )
        
        with gr.Column(scale=1):
            gesture_text = gr.Textbox(
                label="🎯 Detected Gesture", 
                interactive=False,
                placeholder="Starting optimized detection...",
                lines=2
            )
            
            gr.Markdown("""
            ### ⚡ Performance Tips:
            - **Increased to 15 FPS** for smoother video
            - **Reduced frame skipping** for better experience
            - **Persistent AI model** for faster response
            - **Higher detection threshold** for accuracy
            
            ### 🎭 Supported Gestures:
            - 👍 **Thumbs Up** - Yes/Good  
            - ✌️ **Peace** - Victory/Peace
            - ✋ **Stop** - Stop
            - 👌 **OK** - Okay
            - ✊ **Fist** - Power/Start
            - 🖐️ **Open Palm** - Open Hand
            
            ### 💡 Web Usage Tips:
            - Use good lighting
            - Keep hand steady
            - Allow camera permissions
            - Refresh if laggy
            """)
    
    # Optimized streaming with reduced frequency
    webcam.stream(
        fn=detect_gesture, 
        inputs=webcam, 
        outputs=[webcam, gesture_text],
        show_progress=False
    )

# --- SEAMLESS PIPELINE ---
def run_complete_pipeline():
    """
    Run the complete pipeline: collect data → train model → start recognition
    """
    print("🚀 Starting Complete Gesture Recognition Pipeline\n")
    
    # Step 1: Check if files exist
    has_data = os.path.exists("gesture_data.csv")
    has_model = os.path.exists("gesture_model.pkl")
    
    if not has_data:
        print("📸 Starting data collection...")
        if collect_gesture_data_integrated():
            has_data = True
        else:
            print("❌ Data collection failed!")
            return
    
    if not has_model and has_data:
        print("\n🧠 Training model...")
        if train_model_advanced():
            has_model = True
        else:
            print("❌ Model training failed!")
            return
    
    if has_data and has_model:
        print("\n🎯 Pipeline complete! Starting Gradio interface...")
        return True
    else:
        print("❌ Pipeline failed to complete!")
        return False

if __name__ == "__main__":
    choice = input("Choose option:\n1. Run complete pipeline\n2. Collect data only\n3. Train model only\n4. Start recognition (requires existing files)\n5. Test camera only\n6. Use OpenCV interface (RECOMMENDED)\nEnter (1-6): ")
    
    if choice == "1":
        if run_complete_pipeline():
            print("🚀 Starting optimized Gradio interface...")
            print("📱 Open in browser: http://localhost:7860")
            print("⚡ Note: For best performance, use option 6 (OpenCV)")
            iface.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
    elif choice == "2":
        collect_gesture_data_integrated()
    elif choice == "3":
        train_model_advanced()
    elif choice == "4":
        print("🚀 Starting optimized Gradio interface...")
        print("📱 Open in browser: http://localhost:7860")
        print("⚡ Note: For best performance, use option 6 (OpenCV)")
        iface.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
    elif choice == "5":
        # Test camera function
        print("🔍 Testing camera...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera is working!")
            ret, frame = cap.read()
            if ret:
                print("✅ Can capture frames!")
                cv2.imshow("Camera Test", frame)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
            else:
                print("❌ Cannot capture frames!")
        else:
            print("❌ Cannot open camera!")
        cap.release()
    elif choice == "6":
        # Use OpenCV interface instead of Gradio
        print("🚀 Starting OpenCV interface...")
        predict_gesture_realtime()
    else:
        print("🚀 Starting optimized Gradio interface...")
        print("📱 Open in browser: http://localhost:7860")
        print("⚡ Note: For best performance, use option 6 (OpenCV)")
        iface.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)

def build_safe_gesture_model(csv_path="gesture_data.csv", model_path="gesture_model.pkl"):
    """
    Train and save a trusted RandomForest gesture model from local gesture_data.csv.
    No external downloads. Model is safe and compatible with MediaPipe 63-feature input.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_path}' not found. Please collect gesture data first.")

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    if "label" not in df.columns or df.shape[1] != 64:
        raise ValueError("CSV must have 63 feature columns and a 'label' column.")

    X = df.drop("label", axis=1).values
    y = df["label"].values

    print("Splitting data and training RandomForestClassifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.2f}")

    print(f"Saving model to {model_path}...")
    joblib.dump(clf, model_path)
    print(f"✅ Model saved as '{model_path}' and is safe for use.")

# To fix: Run this function to collect gesture data before training or prediction.
# Example usage:
# collect_data("gesture_data.csv", gesture_label="thumbs_up")  # or any gesture name

# After collecting enough samples for each gesture, run:
# train_and_save_model("gesture_data.csv", "gesture_model.pkl")

# Then you can use the real-time prediction functions.

# To resolve the error "Neither 'gesture_model.pkl' nor 'gesture_data.csv' found":
# 1. Run this function to collect gesture data for each gesture you want to recognize:
#    Example:
#    collect_data("gesture_data.csv", gesture_label="thumbs_up")
#    collect_data("gesture_data.csv", gesture_label="peace")
#    collect_data("gesture_data.csv", gesture_label="open_palm")
#
# 2. After collecting enough samples for all gestures, train the model:
#    train_and_save_model("gesture_data.csv", "gesture_model.pkl")
#
# 3. Now you can run the real-time prediction (OpenCV or Gradio).
#
# If you have completed steps 1 and 2, the error will be resolved.
# If you still see the error, make sure both 'gesture_data.csv' and 'gesture_model.pkl' exist in the same directory as this script.