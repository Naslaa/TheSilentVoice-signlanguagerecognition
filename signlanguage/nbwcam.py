import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

# Load the trained CNN model
model = load_model('E:/django/TheSilentVoice-signlanguagerecognition/signlanguage/model/GSNSLmodel.h5')  # Replace with the path to your model
font_path = 'C:/Windows/Fonts/kokila.ttf'

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
            cropped_hand_gray = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            cropped_hand_gray = cv2.resize(cropped_hand_gray, (100, 100))
            cropped_hand_gray = cropped_hand_gray / 255.0
            cropped_hand_gray = np.expand_dims(cropped_hand_gray, axis=0)

            # Make a prediction with the CNN model when space is pressed
            if cv2.waitKey(1) & 0xFF == ord(' '):
                prediction = model.predict(cropped_hand_gray)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]

                # Get the predicted gesture label
                labels = ["क","क्ष", "ख", "ग", "घ", "ङ", "च", "छ", "ज","ज्ञ", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "त्र", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह"]
                gesture_text = labels[predicted_class]


                
                pil_image = Image.fromarray(frame)
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.truetype(font_path, 30)
                draw.text((20, 40), f'{gesture_text} ({confidence:.2f})', font=font, fill=(255, 255, 255))
                frame = np.array(pil_image)


    # if gesture_text is not None and confidence is not None:
    #     cv2.putText(frame, f'{gesture_text} ({confidence:.2f})',
    #                 (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', frame)
 
    # Exit the loop when 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
