# Hand-Gesture
This project is a real-time hand gesture recognition system using OpenCV, deep learning, and hand-tracking modules. It is capable of identifying common gestures like "Hello", "I Love You", "Ok", and "Thank You" from a live webcam feed.

📂 Project Structure
Data.py: Script for capturing and preprocessing hand gesture images.

p.py: Prototype for testing real-time hand gesture classification.

Test.py: Final refined version with custom handling for TensorFlow's DepthwiseConv2D during model loading.

🚀 Features
Hand tracking using cvzone.HandTrackingModule.

Image preprocessing (cropping, resizing, centering).

Gesture classification using a trained TensorFlow/Keras model.

Real-time webcam input and display.

Support for custom-trained models.

🛠️ Requirements
Python 3.x

OpenCV (opencv-python)

cvzone

TensorFlow

NumPy

Install dependencies with:

pip install opencv-python cvzone tensorflow numpy
🖼️ Dataset Creation
Use Data.py to capture hand gesture images:

Press S to save the captured image.

Images are stored in a specified folder for training.

🔍 Usage
1. Collect Gesture Data

python Data.py
Ensure your hand gestures are properly framed when saving images.

2. Real-Time Prediction

python Test.py
Recognizes the gestures.

Displays the prediction and confidence live.

Press S to save a snapshot.

Note: Make sure the model.h5 and labels.txt are correctly placed and the paths are updated if needed.

🧠 Model Info
The model expects centered 300x300 white-background images.

Custom objects like DepthwiseConv2D are properly handled during loading.

📸 Example

Detected Gesture	Model Prediction
Hello	99.5% Confidence
Thank You	97.3% Confidence
📃 Acknowledgments
cvzone for simplifying OpenCV hand tracking.

TensorFlow/Keras for model development.

