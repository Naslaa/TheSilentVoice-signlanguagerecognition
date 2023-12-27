import cv2
import os

# Function to convert images in a directory to black and white
def convert_images_to_bw(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):  # Select file types (adjust as needed)
                try:
                    img_path = os.path.join(root, filename)
                    output_subdir = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, output_subdir)
                    os.makedirs(output_subdir, exist_ok=True)

                    image = cv2.imread(img_path)
                    
                    if image is not None:
                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        output_path = os.path.join(output_subdir, f"bw_{filename}")
                        cv2.imwrite(output_path, gray_image)
                        print(f"Converted {filename} to black and white.")
                    else:
                        print(f"Unable to read {filename}.")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

# Path to your dataset directory containing images
dataset_dir = 'C:/Users/Dell/Downloads/ASLdatasets/'

# Output directory to save black and white images
output_dir = 'C:/Users/Dell/Downloads/GSASLdatasets'

# Convert images in the dataset directory to black and white
convert_images_to_bw(dataset_dir, output_dir)
