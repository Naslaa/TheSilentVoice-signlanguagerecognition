from django.shortcuts import render
from tensorflow.keras.models import load_model
from django.http import JsonResponse
import cv2

from django.views.decorators import gzip

from django.http import HttpResponse
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from django.contrib.staticfiles import finders

import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from django.views.decorators.csrf import csrf_exempt
import cv2



from django.views.decorators import gzip
import cv2

from string import ascii_uppercase
import numpy as np

from django.views import View
from django.http import StreamingHttpResponse
from django.utils.decorators import method_decorator
from django.urls import reverse
import string
import os

from nltk import pos_tag
from django.contrib.staticfiles import finders
from sklearn.feature_extraction.text import CountVectorizer

def find_video(word):
    path = f"C:/django/TheSilentVoice-signlanguagerecognition/signlanguage/static/assets/ASL/{word}.mp4"
    return os.path.isfile(path)


def analyze_text(sentence):
    # Tokenizing the sentence
    words = word_tokenize(sentence.lower())

    # Using NLTK's Part-of-Speech tagging
    tagged = nltk.pos_tag(words)

    stop_words = ['@', '#', "http", ":", "is", "the", "are", "am", "a", "it", "was", "were", "an", ",", ".", "?", "!", ";", "/"]
  
    lr = WordNetLemmatizer()
    filtered_text = []
    for w, p in tagged:
        if w not in stop_words and w not in string.punctuation:
            if p in ['VBG', 'VBD', 'VBZ', 'VBN', 'NN']:
                filtered_text.append(lr.lemmatize(w, pos='v'))
            elif p in ['JJ', 'JJR', 'JJS', 'RBR', 'RBS']:
                filtered_text.append(lr.lemmatize(w, pos='a'))
            else:
                filtered_text.append(w)

    return ' '.join(filtered_text)


# def animation_view(request):
#     if request.method == 'POST':
#         text = request.POST.get('sen')
        
#         analyzed_text = analyze_text(text)
#         analyzed_text_list = [analyzed_text]

#         vectorizer = CountVectorizer()
#         X = vectorizer.fit_transform(analyzed_text_list)

#         vocabulary = vectorizer.get_feature_names_out()

#         words = set(vocabulary)  # Convert vocabulary to a set for faster lookup

#         # Reconstructing the words
#         reconstructed_words = []
#         for word in analyzed_text.split():  
#             if word in words:  
#                 reconstructed_words.append(word)
#             else: 
#                 # If word not found, append individual letters
#                 reconstructed_words.extend(word)

#         return render(request, 'animation.html', {'words': reconstructed_words, 'text': text})
#     else:
#         return render(request, 'animation.html')
def animation_view(request):
    if request.method == 'POST':
        text = request.POST.get('sen')
        
        analyzed_text = analyze_text(text)
        analyzed_text_list = [analyzed_text]

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(analyzed_text_list)

        vocabulary = vectorizer.get_feature_names_out()

        words = set(vocabulary)  # Convert vocabulary to a set for faster lookup

        # Reconstructing the words
        reconstructed_words = []
        for word in analyzed_text.split():  
            if word in words:  
                if find_video(word):  # Check if a video exists for the word
                    reconstructed_words.append(word)
                else:
                    # If video for word is not present, break the word into letters
                    reconstructed_words.extend(word)
            else: 
                # If word not found, append individual letters
                reconstructed_words.extend(word)

        return render(request, 'animation.html', {'words': reconstructed_words, 'text': text})
    else:
        return render(request, 'animation.html')



import unicodedata

def custom_nepali_tokenizer(text):
    tokens = []
    current_token = ''
    
    for char in text:
        if unicodedata.category(char)[0] in ('L', 'M'):  
            current_token += char
        else:
            if current_token:
                tokens.append(current_token)
                current_token = ''
        
            tokens.append(char)

    if current_token:
        tokens.append(current_token)

    return tokens

def get_video_for_letter(letter):
  
    video_path = "C:/django/TheSilentVoice-signlanguagerecognition/signlanguage/static/assets/NSL/{letter}.mp4"  # Adjust this path according to your database structure
    return video_path

def nanimation_view(request):
    if request.method == 'POST':
        text = request.POST.get('sen') 
    
        letters = custom_nepali_tokenizer(text) 

        videos = []
        for letter in letters:
            video_path = get_video_for_letter(letter)
            videos.append(video_path)

        return render(request, 'nanimation.html', {'videos': videos, 'text': text, 'letters':letters})
    else:
        return render(request, 'nanimation.html')


gesture_text = ""
confidence = None
ngesture_text=""
nconfidence = None

model = load_model('C:/django/TheSilentVoice-signlanguagerecognition/signlanguage/model/20GSASLmodel.h5')  #Replace with the path to your model

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils 
def generate_frames():
    global gesture_text, confidence
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
            hand_landmarks = results.multi_hand_landmarks[0]  # Process only the first hand

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

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            cropped_hand = frame[y_min:y_max, x_min:x_max]

            if cropped_hand is not None and cropped_hand.size != 0:
                cropped_hand_gray = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                cropped_hand_gray = cv2.resize(cropped_hand_gray, (100, 100))
                cropped_hand_gray = cropped_hand_gray / 255.0
                cropped_hand_gray = np.expand_dims(cropped_hand_gray, axis=0)
            else:
                print("Error: The cropped_hand image is empty or None.")

            prediction = model.predict(cropped_hand_gray)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]

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

# font_path='C:/Users/Administrator/AppData/Local/Microsoft/Windows/Fonts/kokila.ttf'
nmodel = load_model('model/d7NSLmodel.h5')  # Replace with the path to your model

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils 
def ngenerate_frames():
    global ngesture_text, nconfidence
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
            hand_landmarks = results.multi_hand_landmarks[0]  # Process only the first hand

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

            cropped_hand = frame[y_min:y_max, x_min:x_max]

            if cropped_hand is not None and cropped_hand.size != 0:
                cropped_hand_gray = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                cropped_hand_gray = cv2.resize(cropped_hand_gray, (100, 100))
                cropped_hand_gray = cropped_hand_gray / 255.0
                cropped_hand_gray = np.expand_dims(cropped_hand_gray, axis=0)
            else:
                print("Error: The cropped_hand image is empty or None.")

            prediction = nmodel.predict(cropped_hand_gray)
            predicted_class = np.argmax(prediction)
            nconfidence = prediction[0][predicted_class]

            labels = ["क", "क्ष", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "ज्ञ", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "त्र", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह"]
            ngesture_text = labels[predicted_class]


       
        if ngesture_text is not None and nconfidence is not None:
            text_to_display = f'{ngesture_text} ({nconfidence:.2f})'
        else:
            text_to_display = "No prediction available" 
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        text_font = ImageFont.truetype(font_path, 30)


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
  
    predicted_character = gesture_text


    # Return the predicted data as a JSON response
    return JsonResponse({
        'character': predicted_character,
    },content_type='application/json')

def nperform_prediction(request):
    global ngesture_text

    predicted_character = ngesture_text

    # Return the predicted data as a JSON response
    return JsonResponse({
        'character': predicted_character,
    },content_type='application/json')

def index(request):
    return render(request, 'index.html')
