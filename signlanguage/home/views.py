from django.shortcuts import render, HttpResponse

# Create your views here.


import cv2
from django.http import StreamingHttpResponse
from django.views.decorators import gzip

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

def index(request):
    return render(request, 'index.html')