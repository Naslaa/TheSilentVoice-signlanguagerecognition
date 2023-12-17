import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import pandas as pd
import time

model = load_model('./model/ASLmodel.h5')

# Initialize MediaPipe
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

while True:
    _, frame = cap.read()
    h, w, c = frame.shape

    # Convert BGR frame to RGB for Mediapipe
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20

            # Extract hand region from the frame
            hand_image = frame[y_min:y_max, x_min:x_max]

            # Convert hand image to grayscale
            analysisframe = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
            analysisframe = cv2.resize(analysisframe, (200, 200))  # Adjust as needed for your model

            # Display hand region
            cv2.imshow("Hand Region", analysisframe)

            # Convert image data to DataFrame
            datan = pd.DataFrame(analysisframe)

            # Normalize pixel values
            pixeldata = analysisframe / 255.0

            # Reshape for model prediction
            pixeldata = pixeldata.reshape(-1, 200, 200, 1)

            # Make predictions
            prediction = model.predict(pixeldata)
            predarray = np.array(prediction[0])

            # Print predictions
            letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
            predarrayordered = sorted(predarray, reverse=True)
            high1 = predarrayordered[0]
            high2 = predarrayordered[1]
            high3 = predarrayordered[2]
            for key, value in letter_prediction_dict.items():
                if value == high1:
                    print("Predicted Character 1: ", key)
                    print('Confidence 1: ', 100 * value)
                elif value == high2:
                    print("Predicted Character 2: ", key)
                    print('Confidence 2: ', 100 * value)
                elif value == high3:
                    print("Predicted Character 3: ", key)
                    print('Confidence 3: ', 100 * value)

            time.sleep(5)  # Wait for 5 seconds before capturing another image

    # Draw hand landmarks on the frame
            hand_landmarks = result.multi_hand_landmarks
            if hand_landmarks:
                for handLMs in hand_landmarks:
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                    for lm in handLMs.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    y_min -= 20
                    y_max += 20
                    x_min -= 20
                    x_max += 20
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.imshow("Frame", frame)

    # Exit condition
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
 
cap.release()
cv2.destroyAllWindows()
