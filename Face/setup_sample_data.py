import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

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

def create_sample_gesture_data():
    """
    Create sample gesture data for all 15 deaf-friendly gestures.
    Generates realistic synthetic hand landmark data with gesture-specific patterns.
    """
    print("🔄 Creating sample deaf-friendly gesture data...")
    
    # Use all 15 target gestures
    gestures = list(GESTURE_EMOJI_MAP.keys())
    
    all_data = []
    all_labels = []
    
    print("🎭 Generating synthetic data for gestures:")
    for gesture in gestures:
        emoji = GESTURE_EMOJI_MAP[gesture]
        meaning = GESTURE_MEANINGS[gesture]
        print(f"  {emoji} {gesture} - {meaning}")
    
    # Generate synthetic data for each gesture with more realistic variations
    for gesture in gestures:
        for _ in range(80):  # Increased samples for better diversity
            # Generate 63 features (21 landmarks × 3 coordinates each)
            base_values = np.random.uniform(0.15, 0.85, 63)
            
            # Add gesture-specific realistic patterns with more variation
            if gesture == "thumbs_up":
                base_values[9:12] += np.random.uniform(0.2, 0.4)   # Variable thumb height
                base_values[24:30] -= np.random.uniform(0.1, 0.3)  # Variable finger folding
                base_values[36:42] -= np.random.uniform(0.1, 0.3)
                base_values[48:54] -= np.random.uniform(0.1, 0.3)
                base_values[57:63] -= np.random.uniform(0.1, 0.3)
                
            elif gesture == "thumbs_down":
                base_values[9:12] -= np.random.uniform(0.2, 0.4)   # Variable thumb depth
                base_values[24:30] -= np.random.uniform(0.1, 0.3)
                base_values[36:42] -= np.random.uniform(0.1, 0.3)
                base_values[48:54] -= np.random.uniform(0.1, 0.3)
                base_values[57:63] -= np.random.uniform(0.1, 0.3)
                
            elif gesture == "peace":
                base_values[24:30] += np.random.uniform(0.2, 0.4)  # Variable finger extension
                base_values[36:42] += np.random.uniform(0.2, 0.4)
                base_values[9:15] -= np.random.uniform(0.1, 0.3)
                base_values[48:54] -= np.random.uniform(0.2, 0.4)
                base_values[57:63] -= np.random.uniform(0.2, 0.4)
                
            elif gesture == "i_love_you":
                base_values[9:15] += np.random.uniform(0.15, 0.3)
                base_values[24:30] += np.random.uniform(0.2, 0.4)
                base_values[57:63] += np.random.uniform(0.2, 0.4)
                base_values[36:42] -= np.random.uniform(0.2, 0.4)
                base_values[48:54] -= np.random.uniform(0.2, 0.4)
                
            elif gesture == "wave":
                # Add temporal variation for wave gesture
                wave_variation = np.random.uniform(-0.15, 0.15, 63)
                base_values += wave_variation
                base_values[24:30] += np.random.uniform(0.1, 0.3)
                base_values[36:42] += np.random.uniform(0.1, 0.3)
                base_values[48:54] += np.random.uniform(0.1, 0.3)
                base_values[57:63] += np.random.uniform(0.1, 0.3)
                
            elif gesture == "stop":
                base_values[24:30] += np.random.uniform(0.25, 0.4)
                base_values[36:42] += np.random.uniform(0.25, 0.4)
                base_values[48:54] += np.random.uniform(0.25, 0.4)
                base_values[57:63] += np.random.uniform(0.25, 0.4)
                base_values[9:15] += np.random.uniform(0.15, 0.3)
                
            elif gesture == "okay":
                # Circle formation with variation
                circle_strength = np.random.uniform(0.05, 0.15)
                base_values[9:12] += circle_strength
                base_values[24:27] += circle_strength
                base_values[36:42] += np.random.uniform(0.2, 0.4)
                base_values[48:54] += np.random.uniform(0.2, 0.4)
                base_values[57:63] += np.random.uniform(0.2, 0.4)
                
            elif gesture == "call_me":
                base_values[9:15] += np.random.uniform(0.25, 0.4)
                base_values[57:63] += np.random.uniform(0.25, 0.4)
                base_values[24:30] -= np.random.uniform(0.2, 0.4)
                base_values[36:42] -= np.random.uniform(0.2, 0.4)
                base_values[48:54] -= np.random.uniform(0.2, 0.4)
                
            elif gesture == "fist":
                # Variable compactness
                compactness = np.random.uniform(0.6, 0.8)
                base_values *= compactness
                base_values[24:30] -= np.random.uniform(0.3, 0.5)
                base_values[36:42] -= np.random.uniform(0.3, 0.5)
                base_values[48:54] -= np.random.uniform(0.3, 0.5)
                base_values[57:63] -= np.random.uniform(0.3, 0.5)
                base_values[9:15] -= np.random.uniform(0.2, 0.4)
                
            elif gesture == "open_palm":
                # Variable finger spread
                base_values[24:30] += np.random.uniform(0.3, 0.5)
                base_values[36:42] += np.random.uniform(0.3, 0.5)
                base_values[48:54] += np.random.uniform(0.3, 0.5)
                base_values[57:63] += np.random.uniform(0.3, 0.5)
                base_values[9:15] += np.random.uniform(0.2, 0.4)
                # Add finger spread variation
                spread_variation = np.random.uniform(-0.1, 0.1, 21) * 3
                base_values += np.tile(spread_variation, 3)[:63]
                
            elif gesture == "silence":
                base_values[24:30] += np.random.uniform(0.3, 0.5)
                base_values[36:42] -= np.random.uniform(0.2, 0.4)
                base_values[48:54] -= np.random.uniform(0.2, 0.4)
                base_values[57:63] -= np.random.uniform(0.2, 0.4)
                base_values[9:15] -= np.random.uniform(0.15, 0.3)
                
            elif gesture == "point_right":
                base_values[24:27] += np.random.uniform(0.2, 0.4)
                base_values[27:30] += np.random.uniform(0.15, 0.3)
                base_values[36:42] -= np.random.uniform(0.2, 0.4)
                base_values[48:54] -= np.random.uniform(0.2, 0.4)
                base_values[57:63] -= np.random.uniform(0.2, 0.4)
                
            elif gesture == "point_left":
                base_values[24:27] -= np.random.uniform(0.2, 0.4)
                base_values[27:30] += np.random.uniform(0.15, 0.3)
                base_values[36:42] -= np.random.uniform(0.2, 0.4)
                base_values[48:54] -= np.random.uniform(0.2, 0.4)
                base_values[57:63] -= np.random.uniform(0.2, 0.4)
                
            elif gesture == "point_up":
                base_values[24:30] += np.random.uniform(0.3, 0.5)
                base_values[27] -= np.random.uniform(0.15, 0.3)
                base_values[36:42] -= np.random.uniform(0.2, 0.4)
                base_values[48:54] -= np.random.uniform(0.2, 0.4)
                base_values[57:63] -= np.random.uniform(0.2, 0.4)
                
            elif gesture == "point_down":
                base_values[24:30] += np.random.uniform(0.2, 0.4)
                base_values[27] += np.random.uniform(0.2, 0.4)
                base_values[36:42] -= np.random.uniform(0.2, 0.4)
                base_values[48:54] -= np.random.uniform(0.2, 0.4)
                base_values[57:63] -= np.random.uniform(0.2, 0.4)
            
            # Ensure values stay in valid range [0, 1]
            base_values = np.clip(base_values, 0, 1)
            
            # Add realistic noise and hand tremor simulation
            tremor_noise = np.random.normal(0, 0.015, 63)
            lighting_noise = np.random.normal(0, 0.01, 63)
            camera_noise = np.random.uniform(-0.02, 0.02, 63)
            
            base_values += tremor_noise + lighting_noise + camera_noise
            base_values = np.clip(base_values, 0, 1)
            
            # Add perspective and rotation variation
            perspective_shift = np.random.normal(0, 0.008, 63)
            base_values += perspective_shift
            base_values = np.clip(base_values, 0, 1)
            
            all_data.append(base_values.tolist())
            all_labels.append(gesture)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    df["label"] = all_labels
    
    # Save to CSV
    df.to_csv("gesture_data.csv", index=False)
    print(f"\n✅ Created gesture_data.csv with {len(all_data)} samples")
    print("🎭 Gesture distribution:")
    for gesture, count in df['label'].value_counts().items():
        emoji = GESTURE_EMOJI_MAP.get(gesture, "🤷")
        print(f"  {emoji} {gesture}: {count} samples")
    
    return True

def train_sample_model():
    """
    Train enhanced model with proper train/test split to avoid overfitting.
    """
    print("\n🧠 Training model on sample data...")
    
    # Load data
    df = pd.read_csv("gesture_data.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Proper train/test split to get realistic accuracy
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Enhanced model with regularization to prevent overfitting
    clf = RandomForestClassifier(
        n_estimators=150,           # Reduced to prevent overfitting
        random_state=42,
        max_depth=12,              # Limited depth
        min_samples_split=5,       # Increased to prevent overfitting
        min_samples_leaf=3,        # Added leaf constraint
        class_weight='balanced',
        n_jobs=-1,
        max_features='sqrt'        # Feature subsampling
    )
    clf.fit(X_train, y_train)
    
    # Evaluate on test set
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    
    # Check for overfitting
    overfitting_gap = train_accuracy - test_accuracy
    
    # Save enhanced model package with emoji support
    model_package = {
        'model': clf,
        'gesture_emoji_map': GESTURE_EMOJI_MAP,
        'gesture_meanings': GESTURE_MEANINGS,
        'classes': clf.classes_,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'overfitting_gap': overfitting_gap,
        'version': '2.0_deaf_friendly_robust',
        'total_gestures': len(GESTURE_EMOJI_MAP)
    }
    
    joblib.dump(model_package, "gesture_model.pkl")
    print("✅ Enhanced model saved as gesture_model.pkl")
    print(f"🎯 Train accuracy: {train_accuracy:.3f}")
    print(f"🎯 Test accuracy: {test_accuracy:.3f}")
    print(f"📊 Overfitting gap: {overfitting_gap:.3f}")
    
    if overfitting_gap < 0.05:
        print("✅ Good generalization - low overfitting!")
    elif overfitting_gap < 0.1:
        print("⚠️  Moderate overfitting detected")
    else:
        print("❌ High overfitting - consider more regularization")
    
    print(f"🤟 Supports {len(GESTURE_EMOJI_MAP)} deaf-friendly gestures")
    
    return True

if __name__ == "__main__":
    print("🚀 Setting up deaf-friendly gesture recognition system...")
    print(f"🎯 Target: {len(GESTURE_EMOJI_MAP)} ASL/communication gestures")
    
    # Create sample data and train model
    create_sample_gesture_data()
    train_sample_model()
    
    print("\n🎉 Setup complete!")
    print("🔥 Features:")
    print("  ✅ 15 deaf-friendly gestures with emoji support")
    print("  ✅ Realistic hand landmark patterns")
    print("  ✅ Enhanced RandomForest model")
    print("  ✅ Emoji visualization ready")
    print("\n🎮 You can now run:")
    print("  • main.py - for complete pipeline")
    print("  • app.py - for Gradio interface")
    print("  • Collect real data for better accuracy!")
    
    print("\n🎭 Supported Gestures:")
    for gesture, emoji in GESTURE_EMOJI_MAP.items():
        meaning = GESTURE_MEANINGS[gesture]
        print(f"  {emoji} {gesture} - {meaning}")
