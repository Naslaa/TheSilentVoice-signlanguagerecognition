from django.shortcuts import render, HttpResponse

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


# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# porter = nltk.stem.PorterStemmer()
# wnl = nltk.stem.WordNetLemmatizer()

# #Speech to text
# r = sr.Recognizer()
# mic = sr.Microphone()

# # speech to text
# def speech2text(request):

#     if request.method == 'POST':
#         # Assuming you want to receive the audio file as a POST request
#         audio_file = request.FILES['audio_file']

#         # Initialize the recognizer
#         r = sr.Recognizer()

#         # Adjust for ambient noise
#         r.adjust_for_ambient_noise(audio_file)

#         try:
#             # Listen to the audio file
#             with audio_file as source:
#                 audio = r.listen(source)

#             # Recognize the audio to text
#             text = r.recognize_google(audio)
#             text = text.lower()

#             # Process the text as needed (you can add your processing logic here)
#             processed_text = process_text(text)

#             # Return a JsonResponse with the processed text
#             response_data = {'message': 'Processing complete', 'transcript': processed_text}
#             return JsonResponse(response_data)

#         except sr.UnknownValueError:
#             # Handle the case where no speech could be recognized
#             return JsonResponse({'error': 'Speech recognition failed'}, status=400)
#         except Exception as e:
#             # Handle other exceptions
#             return JsonResponse({'error': str(e)}, status=500)

#     # Handle other HTTP methods or invalid requests
#     return JsonResponse({'error': 'Invalid request'}, status=400)
    
# # def SorT():
# #     speech_or_text=str(input("Type S, if you want to enter speech as input \n else type T if you want to enter text as input.\n"))
# #     if speech_or_text=='s' or speech_or_text == 'S':
# #         speech2text()
# #     elif speech_or_text == 't' or speech_or_text == 'T':
# #         text = str(input("Enter the text:"))
# #     else:
# #         print("Invalid input. Either enter T or S.")
# #         SorT()
# # SorT()

@gzip.gzip_page
def camera_feed(request):
    return StreamingHttpResponse(gen(), content_type="multipart/x-mixed-replace;boundary=frame")

def gen():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Process the frame (perform sign language detection)
        # You can use your sign language detection logic here
        # Example: result_frame = process_sign_language(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def detect_sign_language(request):
    # Perform sign language detection here
    # You can use the captured frames from the camera feed
    return HttpResponse("Sign language detected!")

# def process_text(text):
#     if text.method == 'POST':
#         # Assuming you have a form with a 'text_input' field
#         # text = request.POST.get('text_input', '')

#         # processing the text using bag of words algorithm
#         # Creating a list of stop words or useless words
#         stop = nltk.corpus.stopwords.words('english')
#         stop_words = ['@', '#', "http", ":", "is", "the", "are", "am", "a", "it", "was", "were", "an", ",", ".", "?", "!", ";", "/"]
#         for i in stop_words:
#             stop.append(i)

#         # Processing the text using bag of words
#         tokenized_text = nltk.tokenize.word_tokenize(text)
#         lemmed = [wnl.lemmatize(word) for word in tokenized_text]
#         processed = []
#         for i in lemmed:
#             if i == "i" or i == "I":
#                 processed.append("me")
#             elif i not in stop:
#                 i = i.lower()
#                 processed.append(i)

#         # Pass the processed data to the template
#         context = {
#             'processed_keywords': processed,
#         }
#         return text
#         # return render(request, 'home/index.html', context)

  


# # def Aslanimation(request):
# #     #Showing animation of the keywords.
#     assets_list=['0.mp4', '1.mp4', '2.mp4', '3.mp4', '4.mp4', '5.mp4','6.mp4', '7.mp4', '8.mp4', '9.mp4', 'a.mp4', 'after.mp4',
#                 'again.mp4', 'against.mp4', 'age.mp4', 'all.mp4', 'alone.mp4','also.mp4', 'and.mp4', 'ask.mp4', 'at.mp4', 'b.mp4', 'be.mp4',
#                 'beautiful.mp4', 'before.mp4', 'best.mp4', 'better.mp4', 'busy.mp4', 'but.mp4', 'bye.mp4', 'c.mp4', 'can.mp4', 'cannot.mp4',
#                 'change.mp4', 'college.mp4', 'come.mp4', 'computer.mp4', 'd.mp4', 'day.mp4', 'distance.mp4', 'do not.mp4', 'do.mp4', 'does not.mp4',
#                 'e.mp4', 'eat.mp4', 'engineer.mp4', 'f.mp4', 'fight.mp4', 'finish.mp4', 'from.mp4', 'g.mp4', 'glitter.mp4', 'go.mp4', 'god.mp4',
#                 'gold.mp4', 'good.mp4', 'great.mp4', 'h.mp4', 'hand.mp4', 'hands.mp4', 'happy.mp4', 'hello.mp4', 'help.mp4', 'her.mp4', 'here.mp4',
#                 'his.mp4', 'home.mp4', 'homepage.mp4', 'how.mp4', 'i.mp4', 'invent.mp4', 'it.mp4', 'j.mp4', 'k.mp4', 'keep.mp4', 'l.mp4', 'language.mp4', 'laugh.mp4',
#                 'learn.mp4', 'm.mp4', 'me.mp4', 'mic3.png', 'more.mp4', 'my.mp4', 'n.mp4', 'name.mp4', 'next.mp4', 'not.mp4', 'now.mp4', 'o.mp4', 'of.mp4', 'on.mp4',
#                 'our.mp4', 'out.mp4', 'p.mp4', 'pretty.mp4', 'q.mp4', 'r.mp4', 'right.mp4', 's.mp4', 'sad.mp4', 'safe.mp4', 'see.mp4', 'self.mp4', 'sign.mp4', 'sing.mp4', 
#                 'so.mp4', 'sound.mp4', 'stay.mp4', 'study.mp4', 't.mp4', 'talk.mp4', 'television.mp4', 'thank you.mp4', 'thank.mp4', 'that.mp4', 'they.mp4', 'this.mp4', 'those.mp4', 
#                 'time.mp4', 'to.mp4', 'type.mp4', 'u.mp4', 'us.mp4', 'v.mp4', 'w.mp4', 'walk.mp4', 'wash.mp4', 'way.mp4', 'we.mp4', 'welcome.mp4', 'what.mp4', 'when.mp4', 'where.mp4', 
#                 'which.mp4', 'who.mp4', 'whole.mp4', 'whose.mp4', 'why.mp4', 'will.mp4', 'with.mp4', 'without.mp4', 'words.mp4', 'work.mp4', 'world.mp4', 'wrong.mp4', 'x.mp4', 'y.mp4',
#                 'you.mp4', 'your.mp4', 'yourself.mp4', 'z.mp4']
    
    
#     processed_keywords = context.get('processed_keywords', [])  # Get processed keywords from the context
#     tokens_sign_lan=[]
#     for word in processed_keywords:
#         string = str(word+".mp4")
#         if string in assets_list:
#             tokens_sign_lan.append(str("./ASLvideo/assets/"+string))
#         else:
#             for j in word:
#                 tokens_sign_lan.append(str("./ASLvideo/assets/"+j+".mp4"))

#     return render(text, 'home/index.html')  # Render the form initially

def animation_view(request):
	if request.method == 'POST':
		text = request.POST.get('sen')
		#tokenizing the sentence
		text.lower()
		#tokenizing the sentence
		words = word_tokenize(text)

		tagged = nltk.pos_tag(words)
		tense = {}
		tense["future"] = len([word for word in tagged if word[1] == "MD"])
		tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
		tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]])
		tense["present_continuous"] = len([word for word in tagged if word[1] in ["VBG"]])



		#stopwords that will be removed
		stop_words = set(["mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does', 'shouldn', 'do', "you've",'off', 'for', "didn't", 'm', 'ain', 'haven', "weren't", 'are', "she's", "wasn't", 'its', "haven't", "wouldn't", 'don', 'weren', 's', "you'd", "don't", 'doesn', "hadn't", 'is', 'was', "that'll", "should've", 'a', 'then', 'the', 'mustn', 'i', 'nor', 'as', "it's", "needn't", 'd', 'am', 'have',  'hasn', 'o', "aren't", "you'll", "couldn't", "you're", "mustn't", 'didn', "doesn't", 'll', 'an', 'hadn', 'whom', 'y', "hasn't", 'itself', 'couldn', 'needn', "shan't", 'isn', 'been', 'such', 'shan', "shouldn't", 'aren', 'being', 'were', 'did', 'ma', 't', 'having', 'mightn', 've', "isn't", "won't"])



		#removing stopwords and applying lemmatizing nlp process to words
		lr = WordNetLemmatizer()
		filtered_text = []
		for w,p in zip(words,tagged):
			if w not in stop_words:
				if p[1]=='VBG' or p[1]=='VBD' or p[1]=='VBZ' or p[1]=='VBN' or p[1]=='NN':
					filtered_text.append(lr.lemmatize(w,pos='v'))
				elif p[1]=='JJ' or p[1]=='JJR' or p[1]=='JJS'or p[1]=='RBR' or p[1]=='RBS':
					filtered_text.append(lr.lemmatize(w,pos='a'))

				else:
					filtered_text.append(lr.lemmatize(w))


		#adding the specific word to specify tense
		words = filtered_text
		temp=[]
		for w in words:
			if w=='I':
				temp.append('Me')
			else:
				temp.append(w)
		words = temp
		probable_tense = max(tense,key=tense.get)

		if probable_tense == "past" and tense["past"]>=1:
			temp = ["Before"]
			temp = temp + words
			words = temp
		elif probable_tense == "future" and tense["future"]>=1:
			if "Will" not in words:
					temp = ["Will"]
					temp = temp + words
					words = temp
			else:
				pass
		elif probable_tense == "present":
			if tense["present_continuous"]>=1:
				temp = ["Now"]
				temp = temp + words
				words = temp


		filtered_text = []
		for w in words:
			path = w + ".mp4"
			f = finders.find(path)
			#splitting the word if its animation is not present in database
			if not f:
				for c in w:
					filtered_text.append(c)
			#otherwise animation of word
			else:
				filtered_text.append(w)
		words = filtered_text;


		return render(request,'animation.html',{'words':words,'text':text})
	else:
		return render(request,'animation.html')




def index(request):
    return render(request, 'index.html')