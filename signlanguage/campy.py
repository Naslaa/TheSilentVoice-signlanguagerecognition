import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
# Load the PyTorch model


#CNN Network


class ConvNet(nn.Module):
    def __init__(self,num_classes=29):
        super(ConvNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape= (256,3,150,150)
        
        self.conv1=nn.Conv2d(in_channels=1,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,150,150)
        self.relu1=nn.ReLU()
        #Shape= (256,12,150,150)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,75,75)
        
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,75,75)
        self.relu2=nn.ReLU()
        #Shape= (256,20,75,75)
        
        
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,75,75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,75,75)
        self.relu3=nn.ReLU()
        #Shape= (256,32,75,75)
        
        
        self.fc=nn.Linear(in_features=50 * 50 * 32,out_features=num_classes)
        
        
        
        #Feed forwad function
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
            
            #Above output will be in matrix form, with shape (256,32,75,75)
            
        output=output.view(-1,32*50*50)
            
            
        output=self.fc(output)
            
        return output
            
checkpoint=torch.load('C:/django/TheSilentVoice-signlanguagerecognition/signlanguage/model/best_checkpoint.model')
model=ConvNet(num_classes=29)
model.load_state_dict(checkpoint)
model.eval()
#model = torch.load('C:/django/TheSilentVoice-signlanguagerecognition/signlanguage/model/pytorch_model.model')  # Replace with the path to your PyTorch model

# Set the model to evaluation mode
model.eval()
from torchvision.transforms import transforms
#Transforms
transformer=transforms.Compose([
    transforms.Resize((100,100)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5])
])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Open the webcam

gesture_text = ""
confidence = None

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

            # Prepare the image for prediction
            cropped_hand_normalized = cropped_hand_gray / 255.0
            cropped_hand_tensor = torch.tensor(cropped_hand_normalized, dtype=torch.float32)
            cropped_hand_tensor = cropped_hand_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

            # Make a prediction with the PyTorch model when space is pressed
            if cv2.waitKey(1) & 0xFF == ord(' '):
                with torch.no_grad():
                    output = model(cropped_hand_tensor)
                    predicted_class = torch.argmax(output, dim=1)
                    confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class]

                    # Get the predicted gesture label
                    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "G", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]
                    gesture_text = labels[predicted_class]

    # if gesture_text != "" and confidence is not None:
    #     cv2.putText(frame, f'{gesture_text} ({confidence:.2f})',
    #                 (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    if gesture_text != "" and confidence is not None:
    # Convert tensor to Python float for formatting
        confidence_value = confidence.item()
        cv2.putText(frame, f'{gesture_text} ({confidence_value:.2f})',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit the loop when 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
