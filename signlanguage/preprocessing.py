import cv2
import os
from PIL import Image

# Function to extract frames, resize to 100x100, and convert to black and white
def extract_frames(input_folder, output_folder, max_frames=35):
    for root, _, files in os.walk(input_folder):
        for file in files:
            video_path = os.path.join(root, file)

            # Create output subfolder based on video file
            video_name = os.path.splitext(file)[0]
            output_subfolder = os.path.join(output_folder, video_name)
            os.makedirs(output_subfolder, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                current_frame = 0
                while current_frame < max_frames:
                    ret, frame = cap.read()
                    if ret:
                        # Convert frame to black and white
                        bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # Resize the frame to 100x100 pixels
                        resized_frame = cv2.resize(bw_frame, (100, 100))

                        # Save the frame
                        frame_name = f'{output_subfolder}/frame_{current_frame}.jpg'
                        cv2.imwrite(frame_name, resized_frame)
                        current_frame += 1
                    else:
                        break  # Break the loop if there are no more frames

                cap.release()
            else:
                print(f"Error: Unable to open video file at {video_path}")

# Replace 'input_folder' with the path to your input videos folder
input_folder = '/content/drive/MyDrive/crop nsl college video/'

# Replace 'output_folder' with the path to your output frames folder
output_folder = '/content/drive/MyDrive/college video output blackandwhite/'

extract_frames(input_folder, output_folder, max_frames=35)