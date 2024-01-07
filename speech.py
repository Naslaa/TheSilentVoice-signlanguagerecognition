import pyttsx3

engine = pyttsx3.init()
engine.say("H")
engine.runAndWait()
engine.stop()