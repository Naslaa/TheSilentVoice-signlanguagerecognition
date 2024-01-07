from django.shortcuts import render
from tensorflow.keras.models import load_model


import cv2
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from django.http import JsonResponse
import nltk
import cv2
import time
import speech_recognition as sr
import pyttsx3
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from django.contrib.staticfiles import finders

import mediapipe as mp
import numpy as np
import tensorflow as tf
from django.http import StreamingHttpResponse
import time
from django.views.decorators.csrf import csrf_exempt
import cv2



# In your Django app's views.py
from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import cv2

from string import ascii_uppercase




# 
import cv2
import numpy as np
import base64


from django.views.decorators import gzip
from django.views import View
from django.http import StreamingHttpResponse
from django.utils.decorators import method_decorator
from django.urls import reverse

import os
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from django.contrib.staticfiles import finders
from sklearn.feature_extraction.text import CountVectorizer

def find_video(word):
    path = f"E:/django/TheSilentVoice-signlanguagerecognition/signlanguage/static/assets/ASL/{word}.mp4"  # Change 'path_to_video_folder' to your video folder path
    return os.path.isfile(path)

# Function to analyze text using Bag of Words model
def analyze_text(sentence):
    # Tokenizing the sentence
    words = word_tokenize(sentence.lower())

    # Using NLTK's Part-of-Speech tagging
    tagged = nltk.pos_tag(words)

    # Lemmatizing and filtering words
    lr = WordNetLemmatizer()
    filtered_text = []
    for w, p in zip(words, tagged):
        if p[1] in ['VBG', 'VBD', 'VBZ', 'VBN', 'NN']:
            filtered_text.append(lr.lemmatize(w, pos='v'))
        elif p[1] in ['JJ', 'JJR', 'JJS', 'RBR', 'RBS']:
            filtered_text.append(lr.lemmatize(w, pos='a'))
        else:
            filtered_text.append(lr.lemmatize(w))

    return ' '.join(filtered_text)

def animation_view(request):
    if request.method == 'POST':
        text = request.POST.get('sen')
        
        # Analyze the text using the Bag of Words model
        analyzed_text = analyze_text(text)

        # Create a list with the analyzed text
        analyzed_text_list = [analyzed_text]

        # Using CountVectorizer to create Bag of Words representation
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(analyzed_text_list)

        # Get the vocabulary
        vocabulary = vectorizer.get_feature_names_out()

        # List of words from the analyzed text
        words = vocabulary.tolist()

        # Check for existence of videos in the database
        filtered_text = []
        for w in words:
            # Check if video for word exists in the database
            if find_video(w):
                filtered_text.append(w)
            else:
                # If video for the word doesn't exist, play videos of individual letters
                for c in w:
                    filtered_text.append(c)

        words = filtered_text

        return render(request, 'animation.html', {'words': words, 'text': text})
    else:
        return render(request, 'animation.html')

import unicodedata

def custom_nepali_tokenizer(text):
    tokens = []
    current_token = ''
    
    for char in text:
        if unicodedata.category(char)[0] in ('L', 'M'):  # Checks if the character is a letter or a combining mark
            current_token += char
        else:
            if current_token:
                tokens.append(current_token)
                current_token = ''
            # Append non-letter characters as tokens
            tokens.append(char)

    if current_token:
        tokens.append(current_token)

    return tokens

def get_video_for_letter(letter):
    # Replace this function with code to retrieve the video for a Nepali letter from the database
    # This function should return the path or URL of the video file corresponding to the letter
    # Example:
    video_path = f'E:/django/TheSilentVoice-signlanguagerecognition/signlanguage/static/assets/NSL/{letter}.mp4'  # Adjust this path according to your database structure
    return video_path

def nanimation_view(request):
    if request.method == 'POST':
        text = request.POST.get('sen')
        
        # Perform necessary Nepali language processing steps here
        letters = custom_nepali_tokenizer(text)  # Break Nepali text into individual letters

        # Extract videos for each letter
        videos = []
        for letter in letters:
            video_path = get_video_for_letter(letter)
            videos.append(video_path)

        return render(request, 'nanimation.html', {'videos': videos, 'text': text, 'letters':letters})
    else:
        return render(request, 'nanimation.html')


# def camera_feed(request):
# 	result= subprocess.run(['python', './bwcam.py'], capture_output=True, text=True)
# 	return render(request, 'camera-feed.html',{'output':result.stdout})

# def ncamera_feed(request):
#     result = subprocess.run(['python', './nbwcam.py'], capture_output=True, text=True)
#     output = result.stdout
#     return render(request, 'ncamera-feed.html', {'output': output})
model = load_model('C:/django/TheSilentVoice-signlanguagerecognition/signlanguage/model/20GSASLmodel.h5')  # Replace with the path to your model

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils 
def generate_frames():
    global gesture_text, confidence
    cap = cv2.VideoCapture(0)  # Open the webcam
    prediction = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
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
                # if cv2.waitKey(1) & 0xFF == ord(' '):
                prediction = model.predict(cropped_hand_gray)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]

                    # Get the predicted gesture label
                labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "G", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]
                gesture_text = labels[predicted_class]

        if gesture_text is not None and confidence is not None:
            cv2.putText(frame, f'{gesture_text} ({confidence:.2f})',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def camera_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def camera_view(request):
    return render(request, 'camera-feed.html')

from PIL import Image, ImageDraw, ImageFont
font_path = 'c:/Windows/Fonts/kokila.ttf'
nmodel = load_model('C:/django/TheSilentVoice-signlanguagerecognition/signlanguage/model/d7NSLmodel.h5')  # Replace with the path to your model

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils 
def ngenerate_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam

    prediction = None
    confidence = None
    gesture_text = "" 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
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
                # if cv2.waitKey(1) & 0xFF == ord(' '):
                prediction = nmodel.predict(cropped_hand_gray)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]

                    # Get the predicted gesture label
                labels = ["क","क्ष", "ख", "ग", "घ", "ङ", "च", "छ", "ज","ज्ञ", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "त्र", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह"]
                gesture_text = labels[predicted_class]

        # if gesture_text is not None and confidence is not None:
        #     cv2.putText(frame, f'{gesture_text} ({confidence:.2f})',
        #                 (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
       
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        text_font = ImageFont.truetype(font_path, 30)

        if gesture_text is not None and confidence is not None:
            text_to_display = f'{gesture_text} ({confidence:.2f})'
        else:
            text_to_display = "No prediction available"  # Placeholder text if either is None

        draw.text((20, 40), text_to_display, font=text_font, fill=(255, 255, 255))
        frame = np.array(pil_image)


        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def ncamera_feed(request):
    return StreamingHttpResponse(ngenerate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def ncamera_view(request):
    return render(request, 'ncamera-feed.html')

def perform_prediction(request):
    global gesture_text
    # Add code to perform the prediction using your model
    # Example placeholder response (modify as per your actual prediction logic)
    predicted_character = gesture_text
    predicted_word = "Apple"
    predicted_sentence = "This is a sample sentence."

    # Return the predicted data as a JSON response
    return JsonResponse({
        'character': predicted_character,
        'word': predicted_word,
        'sentence': predicted_sentence,
    },content_type='application/json')

def index(request):
    return render(request, 'index.html')

# def ncamera_feed(request):
# 	return render(request, 'ncamera-feed.html')