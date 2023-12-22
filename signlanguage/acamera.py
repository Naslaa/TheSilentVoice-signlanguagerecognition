import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import unicodedata

# Load the trained CNN model
model = load_model('C:/django/TheSilentVoice-signlanguagerecognition/signlanguage/model/NSLmodel.h5')  # Replace with the path to your model

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Open the webcam

prediction = None
confidence = None
gesture_text = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB and process it with MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand landmarks
            h, w, c = frame.shape
            x_max, y_max, x_min, y_min = 0, 0, w, h
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

            # Draw a rectangle around the detected hand region
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Crop the hand region for prediction
            cropped_hand = frame[y_min:y_max, x_min:x_max]
            cropped_hand = cv2.resize(cropped_hand, (100, 100))
            cropped_hand = cropped_hand / 255.0
            cropped_hand = np.expand_dims(cropped_hand, axis=0)

            # Make a prediction with the CNN model when space is pressed
            if cv2.waitKey(1) & 0xFF == ord(' '):
                prediction = model.predict(cropped_hand)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]

                # Get the predicted gesture label
                # Replace this list with your labels for different gestures
                labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "G", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

                gesture_text = labels[predicted_class]

    # cv2.putText(frame, f'{gesture_text} ({confidence:.2f})',
    #             (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    if gesture_text is not None and confidence is not None:
        cv2.putText(frame, f'{gesture_text} ({confidence:.2f})',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit the loop when 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
