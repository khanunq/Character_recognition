import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load the pre-trained model
with open('model.p', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']

# Define the label names
label_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X']

# Initialize MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(frame_rgb)

    # Check if hand(s) detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box coordinates around the hand
            x_min = int(min(hand_landmarks.landmark, key=lambda x: x.x).x * frame.shape[1])
            y_min = int(min(hand_landmarks.landmark, key=lambda x: x.y).y * frame.shape[0])
            x_max = int(max(hand_landmarks.landmark, key=lambda x: x.x).x * frame.shape[1])
            y_max = int(max(hand_landmarks.landmark, key=lambda x: x.y).y * frame.shape[0])

            # Draw rectangle around the hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)
    gray = gray / 255.0

    # Make a prediction using the model
    prediction = model.predict(gray)
    predicted_label = label_names[np.argmax(prediction)]

    # Draw the predicted label on the frame
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
