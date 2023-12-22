import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import time

# Load the trained CNN model
model = load_model('C:/django/TheSilentVoice-signlanguagerecognition/signlanguage/model/NSLmodel.h5')
font_path = 'c:/Windows/Fonts/kokila.ttf'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Open the webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
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

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            cropped_hand = frame[y_min:y_max, x_min:x_max]
            cropped_hand = cv2.resize(cropped_hand, (100, 100))
            cropped_hand = cropped_hand / 255.0
            cropped_hand = np.expand_dims(cropped_hand, axis=0)

            if cv2.waitKey(1) & 0xFF == ord(' '):
                prediction = model.predict(cropped_hand)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]

                labels = ["क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह", "क्ष", "त्र", "ज्ञ"]
                gesture_text = labels[predicted_class]

                pil_image = Image.fromarray(frame)
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.truetype(font_path, 30)
                draw.text((20, 40), f'{gesture_text} ({confidence:.2f})', font=font, fill=(255, 255, 255))
                frame = np.array(pil_image)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) == 27:  # Exit when Esc key is pressed
        break

cap.release()
cv2.destroyAllWindows()
