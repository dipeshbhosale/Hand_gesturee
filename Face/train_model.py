import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 🎯 DEAF-FRIENDLY GESTURE EMOJI MAPPING
GESTURE_EMOJI_MAP = {
    "thumbs_up": "👍",      # Yes / Good
    "thumbs_down": "👎",    # No / Bad
    "peace": "✌️",          # Victory / Peace
    "i_love_you": "🤟",     # ASL 'I Love You'
    "wave": "👋",           # Hello / Bye
    "stop": "✋",           # Stop
    "okay": "👌",          # OK
    "call_me": "🤙",       # Call me / Aloha
    "fist": "✊",           # Power / Start
    "open_palm": "🖐️",     # Open hand
    "silence": "🤫",       # Quiet
    "point_right": "👉",   # Direction
    "point_left": "👈",    # Direction
    "point_up": "☝️",      # Attention
    "point_down": "👇"     # Look below
}

GESTURE_MEANINGS = {
    "thumbs_up": "Yes / Good",
    "thumbs_down": "No / Bad", 
    "peace": "Victory / Peace",
    "i_love_you": "ASL 'I Love You'",
    "wave": "Hello / Bye",
    "stop": "Stop",
    "okay": "OK",
    "call_me": "Call me / Aloha",
    "fist": "Power / Start",
    "open_palm": "Open hand",
    "silence": "Quiet",
    "point_right": "Direction",
    "point_left": "Direction", 
    "point_up": "Attention",
    "point_down": "Look below"
}

def advanced_data_augmentation(X, y, augment_factor=2):
    """
    Enhanced data augmentation for deaf-friendly gestures.
    Includes hand orientation variations and lighting conditions simulation.
    """
    X_aug = []
    y_aug = []
    
    for xi, yi in zip(X, y):
        # Original sample
        X_aug.append(xi)
        y_aug.append(yi)
        
        for _ in range(augment_factor):
            landmarks = xi.reshape(21, 3)
            
            # 1. Gaussian noise (camera shake/lighting variation)
            noise = np.random.normal(0, 0.002, landmarks.shape)
            noisy_landmarks = landmarks + noise
            
            # 2. Scaling (distance from camera)
            scale_factor = np.random.uniform(0.92, 1.08)
            scaled_landmarks = landmarks * scale_factor
            
            # 3. Translation (hand position shift)
            translation = np.random.normal(0, 0.005, (1, 3))
            translated_landmarks = landmarks + translation
            
            # 4. Rotation simulation (slight hand tilt)
            if np.random.random() > 0.6:
                rotation_angle = np.random.uniform(-0.1, 0.1)
                cos_theta, sin_theta = np.cos(rotation_angle), np.sin(rotation_angle)
                rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
                
                rotated_landmarks = landmarks.copy()
                xy_coords = landmarks[:, :2]
                rotated_xy = np.dot(xy_coords, rotation_matrix.T)
                rotated_landmarks[:, :2] = rotated_xy
                
                X_aug.append(np.clip(rotated_landmarks.flatten(), 0, 1))
                y_aug.append(yi)
            
            # 5. Horizontal flip (for symmetric gestures)
            if yi in ["wave", "open_palm", "stop", "silence"] and np.random.random() > 0.7:
                mirror_landmarks = landmarks.copy()
                mirror_landmarks[:, 0] = 1 - mirror_landmarks[:, 0]
                X_aug.append(np.clip(mirror_landmarks.flatten(), 0, 1))
                y_aug.append(yi)
            
            # Add other augmented samples
            for aug_landmarks in [noisy_landmarks, scaled_landmarks, translated_landmarks]:
                aug_landmarks = np.clip(aug_landmarks, 0, 1)
                X_aug.append(aug_landmarks.flatten())
                y_aug.append(yi)
    
    return np.array(X_aug), np.array(y_aug)

def create_deaf_gesture_features(X):
    """
    Create specialized features for deaf-friendly gesture recognition.
    Based on ASL research and hand biomechanics.
    """
    X_features = []
    
    for sample in X:
        landmarks = sample.reshape(21, 3)
        features = list(sample)  # Original 63 features
        
        # Key landmarks for ASL/deaf gestures
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        # 1. FINGER EXTENSION FEATURES (critical for ASL)
        def finger_extension_ratio(tip, pip):
            tip_dist = np.linalg.norm(tip - wrist)
            pip_dist = np.linalg.norm(pip - wrist)
            return tip_dist / (pip_dist + 1e-8)
        
        features.extend([
            finger_extension_ratio(thumb_tip, thumb_mcp),    # Thumb extension
            finger_extension_ratio(index_tip, index_pip),    # Index extension
            finger_extension_ratio(middle_tip, middle_pip),  # Middle extension
            finger_extension_ratio(ring_tip, ring_pip),      # Ring extension
            finger_extension_ratio(pinky_tip, pinky_pip)     # Pinky extension
        ])
        
        # 2. HAND ORIENTATION FEATURES
        # Palm normal vector (important for orientation-based gestures)
        palm_points = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
        palm_center = np.mean(palm_points, axis=0)
        
        # Vector from wrist to middle finger base
        hand_direction = landmarks[9] - wrist
        hand_direction_norm = hand_direction / (np.linalg.norm(hand_direction) + 1e-8)
        features.extend(hand_direction_norm)
        
        # 3. FINGER SPREAD FEATURES (for open_palm, stop, etc.)
        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        finger_spreads = []
        for i in range(len(finger_tips)-1):
            spread = np.linalg.norm(finger_tips[i] - finger_tips[i+1])
            finger_spreads.append(spread)
        features.extend(finger_spreads)
        
        # 4. THUMB POSITION FEATURES (critical for thumbs_up/down, okay, etc.)
        # Thumb relative to index
        thumb_to_index = thumb_tip - index_tip
        features.extend(thumb_to_index)
        
        # Thumb relative to palm center
        thumb_to_palm = thumb_tip - palm_center
        features.extend(thumb_to_palm)
        
        # 5. FIST/OPEN DETECTION FEATURES
        # Average distance of fingertips from palm
        fingertip_distances = [np.linalg.norm(tip - palm_center) for tip in finger_tips + [thumb_tip]]
        features.extend([
            np.mean(fingertip_distances),  # Average fingertip distance
            np.std(fingertip_distances),   # Fingertip distance variation
            min(fingertip_distances),      # Closest fingertip
            max(fingertip_distances)       # Farthest fingertip
        ])
        
        # 6. POINTING DIRECTION FEATURES
        # Index finger pointing direction
        index_direction = index_tip - index_pip
        index_direction_norm = index_direction / (np.linalg.norm(index_direction) + 1e-8)
        features.extend(index_direction_norm)
        
        # 7. ASL-SPECIFIC FEATURES
        # "I Love You" gesture: thumb, index, and pinky extended
        ily_score = (
            finger_extension_ratio(thumb_tip, thumb_mcp) +
            finger_extension_ratio(index_tip, index_pip) +
            finger_extension_ratio(pinky_tip, pinky_pip) -
            finger_extension_ratio(middle_tip, middle_pip) -
            finger_extension_ratio(ring_tip, ring_pip)
        )
        features.append(ily_score)
        
        # "OK" gesture: thumb-index circle
        thumb_index_distance = np.linalg.norm(thumb_tip - index_tip)
        features.append(thumb_index_distance)
        
        # "Peace" gesture: index and middle extended, others folded
        peace_score = (
            finger_extension_ratio(index_tip, index_pip) +
            finger_extension_ratio(middle_tip, middle_pip) -
            finger_extension_ratio(ring_tip, ring_pip) -
            finger_extension_ratio(pinky_tip, pinky_pip)
        )
        features.append(peace_score)
        
        X_features.append(features)
    
    return np.array(X_features)

def train_gesture_model():
    """
    Train advanced deaf-friendly gesture recognition model.
    Optimized for 15 ASL/communication gestures with emoji support.
    """
    # Check if data file exists
    if not os.path.exists("gesture_data.csv"):
        print("❌ gesture_data.csv not found!")
        print("Run data collection first to collect training data for these gestures:")
        print("\n🎯 TARGET GESTURES:")
        for gesture, emoji in GESTURE_EMOJI_MAP.items():
            meaning = GESTURE_MEANINGS[gesture]
            print(f"  {emoji} {gesture} - {meaning}")
        return
    
    print("📊 Loading deaf-friendly gesture data...")
    df = pd.read_csv("gesture_data.csv")
    
    # Validate data format
    if "label" not in df.columns:
        print("❌ Invalid CSV format: 'label' column missing")
        return
    
    if df.shape[1] != 64:  # 63 features + 1 label
        print(f"❌ Invalid CSV format: Expected 64 columns, got {df.shape[1]}")
        return
    
    print(f"✅ Loaded {len(df)} samples with {len(df['label'].unique())} gestures")
    
    # Show gesture distribution with emojis
    print("\n🎭 Gesture Distribution:")
    for gesture, count in df['label'].value_counts().items():
        emoji = GESTURE_EMOJI_MAP.get(gesture, "🤷")
        meaning = GESTURE_MEANINGS.get(gesture, "Unknown")
        print(f"  {emoji} {gesture}: {count} samples - {meaning}")
    
    # Prepare features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Advanced deaf-gesture feature engineering
    print("\n🔧 Creating deaf-gesture specialized features...")
    X_advanced = create_deaf_gesture_features(X)
    print(f"Feature dimensions: {X.shape[1]} → {X_advanced.shape[1]}")
    
    # Enhanced data augmentation
    print("🔄 Applying deaf-gesture data augmentation...")
    X_aug, y_aug = advanced_data_augmentation(X_advanced, y, augment_factor=2)
    print(f"Dataset size: {len(X_advanced)} → {len(X_aug)} samples")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )
    
    # Feature scaling for better convergence
    print("⚡ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection for speed and memory efficiency
    print("🎯 Selecting best features...")
    selector = SelectKBest(score_func=f_classif, k=min(80, X_train_scaled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"Selected {X_train_selected.shape[1]} best features")
    
    # Specialized models for gesture recognition
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            subsample=0.9
        ),
        'SVC': SVC(
            C=100,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            random_state=42,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    
    # Train and evaluate models with cross-validation
    print(f"\n🔄 Training ensemble models on {len(X_train_selected)} samples...")
    
    best_models = {}
    cv_scores = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        
        # Train model
        model.fit(X_train_selected, y_train)
        
        # Cross-validation
        cv_score = cross_val_score(model, X_train_selected, y_train, cv=skf, scoring='accuracy')
        cv_scores[name] = cv_score
        
        # Test accuracy
        test_score = model.score(X_test_selected, y_test)
        
        print(f"  CV Score: {cv_score.mean():.3f} (±{cv_score.std()*2:.3f})")
        print(f"  Test Score: {test_score:.3f}")
        
        best_models[name] = model
    
    # Create voting ensemble for maximum accuracy
    print("\n🎭 Creating voting ensemble...")
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in best_models.items()],
        voting='soft'
    )
    ensemble.fit(X_train_selected, y_train)
    
    # Evaluate ensemble
    ensemble_cv = cross_val_score(ensemble, X_train_selected, y_train, cv=skf, scoring='accuracy')
    ensemble_test = ensemble.score(X_test_selected, y_test)
    
    print(f"Ensemble CV Score: {ensemble_cv.mean():.3f} (±{ensemble_cv.std()*2:.3f})")
    print(f"Ensemble Test Score: {ensemble_test:.3f}")
    
    # Select best performing model
    all_scores = {name: model.score(X_test_selected, y_test) for name, model in best_models.items()}
    all_scores['Ensemble'] = ensemble_test
    
    best_model_name = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_model_name]
    
    if best_model_name == 'Ensemble':
        final_model = ensemble
    else:
        final_model = best_models[best_model_name]
    
    # Create optimized pipeline for deployment
    final_pipeline = Pipeline([
        ('feature_engineering', 'passthrough'),  # Will be handled in preprocessing
        ('scaler', scaler),
        ('selector', selector),
        ('classifier', final_model)
    ])
    
    # Refit on training data
    final_pipeline.fit(X_train, y_train)
    final_test_score = final_pipeline.score(X_test, y_test)
    
    # Detailed evaluation
    y_pred = final_pipeline.predict(X_test)
    
    print(f"\n📈 Final Model Performance ({best_model_name}):")
    print(f"Accuracy: {final_test_score:.3f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance (if available)
    if hasattr(final_model, 'feature_importances_'):
        feature_importance = final_model.feature_importances_
        print(f"\nTop 5 most important features:")
        top_indices = np.argsort(feature_importance)[-5:][::-1]
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.3f}")
    
    # Memory optimization: save only essential components
    model_package = {
        'pipeline': final_pipeline,
        'model_name': best_model_name,
        'accuracy': final_test_score,
        'gesture_emoji_map': GESTURE_EMOJI_MAP,
        'gesture_meanings': GESTURE_MEANINGS,
        'feature_names': [f'feature_{i}' for i in range(X_advanced.shape[1])],
        'classes': final_pipeline.classes_,
        'target_gestures': list(GESTURE_EMOJI_MAP.keys()),
        'version': '2.0_deaf_friendly'
    }
    
    # Save optimized model
    joblib.dump(model_package, "gesture_model.pkl", compress=3)
    
    print(f"\n✅ Deaf-friendly gesture model saved as gesture_model.pkl")
    print(f"🎯 Final Accuracy: {final_test_score:.3f}")
    print(f"🚀 Model: {best_model_name}")
    print(f"🤟 Supports {len(GESTURE_EMOJI_MAP)} ASL/deaf-friendly gestures")
    print("\n🎭 Supported Gestures:")
    for gesture, emoji in GESTURE_EMOJI_MAP.items():
        if gesture in final_pipeline.classes_:
            meaning = GESTURE_MEANINGS[gesture]
            print(f"  {emoji} {gesture} - {meaning}")
    
    print("\n🔥 Enhanced with:")
    print("  ✅ ASL-specific features")
    print("  ✅ Deaf-community gestures")  
    print("  ✅ Emoji visualization support")
    print("  ✅ Advanced data augmentation")
    print("  ✅ Multi-model ensemble")
    print("⚡ Ready for real-time deaf-friendly communication!")

if __name__ == "__main__":
    train_gesture_model()
