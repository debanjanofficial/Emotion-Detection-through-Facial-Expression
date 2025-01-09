import cv2
import torch
import numpy as np
from torchvision import transforms
from src.model import EmotionNetResNet50
from src.config import Config

# Define emotion labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Define the to_rgb function to convert grayscale images to RGB
def to_rgb(image):
    """Convert grayscale image to 3-channel RGB."""
    return image.repeat(3, 1, 1)

# Define transformations for preprocessing the image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),  # Resize to match model input size
    transforms.ToTensor(),       # Convert to PyTorch tensor
    transforms.Lambda(to_rgb),   # Convert grayscale to RGB
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load the trained model
model = EmotionNetResNet50()
model.load_state_dict(torch.load(Config.save_path, map_location=Config.device))
model.to(Config.device)
model.eval()

# Load Haar Cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_haar_cascade.empty():
    raise RuntimeError("Error loading Haar Cascade XML file for face detection.")

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Could not access webcam. Check if it's being used by another application.")

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame. Retrying...")
        continue

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop and preprocess the face
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_tensor = transform(roi_resized).unsqueeze(0).to(Config.device)

        # Make prediction
        with torch.no_grad():
            predictions = model(roi_tensor)
            max_index = torch.argmax(predictions[0]).item()
            confidence = torch.softmax(predictions[0], dim=0)[max_index].item()
            predicted_emotion = f"{emotions[max_index]} ({confidence:.2f})"

        # Display predicted emotion and confidence
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    resized_frame = cv2.resize(frame, (1000, 700))
    cv2.imshow('Facial Emotion Detection', resized_frame)

    # Exit on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()