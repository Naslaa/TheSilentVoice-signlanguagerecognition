from django.shortcuts import render

# Create your views here.


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
import base64
from io import BytesIO
import subprocess


# In your Django app's views.py
from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import cv2
import operator
from string import ascii_uppercase
from spellchecker import SpellChecker
from keras.models import model_from_json


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
    path = f"C:/django/TheSilentVoice-signlanguagerecognition/signlanguage/static/assets/ASL/{word}.mp4"  # Change 'path_to_video_folder' to your video folder path
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
    video_path = f'static/assets/NSL/{letter}.mp4'  # Adjust this path according to your database structure
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

        return render(request, 'nanimation.html', {'videos': videos, 'text': text})
    else:
        return render(request, 'nanimation.html')


def camera_feed(request):
	result= subprocess.run(['python', './camera.py'], capture_output=True, text=True)
	return render(request, 'camera-feed.html',{'output':result.stdout})


def index(request):
    return render(request, 'index.html')

def ncamera_feed(request):
	return render(request, 'ncamera-feed.html')