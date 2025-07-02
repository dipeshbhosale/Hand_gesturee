# Hand Gesture Recognition System

## Quick Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Collect gesture data:**
   ```bash
   python collect_gestures.py
   ```
   - Follow on-screen instructions
   - Collect 50+ samples per gesture
   - Press SPACE to capture, Q to move to next gesture

3. **Train the model:**
   ```bash
   python train_model.py
   ```
   - Creates gesture_model.pkl from your collected data

4. **Run real-time recognition:**
   ```bash
   python main.py
   ```
   - Opens Gradio web interface for live gesture recognition

## Files Created
- `gesture_data.csv` - Your collected training data
- `gesture_model.pkl` - Trained model for recognition
