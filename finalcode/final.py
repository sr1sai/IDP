import cv2
import numpy as np
import os
import time
import torch
import torchvision.transforms as transforms
from PIL import Image

def extractor(vid_path):
    
    cap = cv2.VideoCapture(vid_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Initialize a list to store unique frames
    frames = []

    # Initialize a variable to keep track of frame numbers
    frame_count = 0

    # Read the first frame
    ret, prev_frame = cap.read()

    # Convert the frame to a grayscale image for faster comparison
    #prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frames.append(prev_frame)

    while True:
        ret, frame = cap.read()

        # Break the loop when we reach the end of the video
        if not ret:
            break

        # Convert the frame to grayscale for comparison
        #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the absolute difference between the current and previous frames
        # frame_diff = cv2.absdiff(prev_frame_gray, frame_gray)
        frame_diff = cv2.absdiff(prev_frame, frame)
        # If the frame is not a duplicate (based on a simple threshold),
        # add it to the list of frames and update the previous frame
        if np.mean(frame_diff) > 5:
            frames.append(frame)
        #  prev_frame_gray = frame_gray.copy()
            prev_frame = frame.copy()
            
        # Display the frame (optional)
        # cv2.imshow('Frame', frame_gray)
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Release the video capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()
    print("no of frames: ",len(frames))

    video_name = os.path.basename(vid_path).split('.')[0]

    # frames_dict= { frame_name:frame 

    frames_dict={}
    
    for i, frame in enumerate(frames):
        frames_dict[video_name+"_frame_"+str(i)]=frame
    
    return frames_dict


def check_image_high_resolution(image):
    width, height = image.size
    threshold_width = 2048  # Define the threshold width for high resolution
    threshold_height = 2048  # Define the threshold height for high resolution
    if width >= threshold_width or height >= threshold_height:
        print("Image is already at a high resolution, skipping upscaling.")
        return True
    return False

def enhance_photos(photo_dict):
    enhanced_photos = {}
    model_path = 'C:\Users\Sri Sai\OneDrive\Desktop\IDP\ESRGAN\models\RRDB_ESRGAN_x4.pth'
    model = torch.load(model_path)

    for photo_name, photo in photo_dict.items():
        input_image = photo

        if check_image_high_resolution(input_image):
            enhanced_photos[photo_name] = input_image
            continue  # Skip processing if the image is already at a high resolution

        preprocess = transforms.Compose([
            transforms.ToTensor()    # Convert to tensor
        ])
        input_tensor = preprocess(input_image).unsqueeze(0)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
        enhanced_photos[photo_name] = output_image

    return enhanced_photos

vid_path="./finalcode/vid3.mp4"
frames=extractor(vid_path)
print(frames.keys())


#checking the frames
"""for frame_name, frame in frames.items():
    cv2.imshow(frame_name, frame)
    cv2.waitKey(0)  # Display each frame until any key is pressed
    cv2.destroyWindow(frame_name)  # Close the frame window after viewing

cv2.waitKey(0)  # Wait for any key to be pressed to close all frame windows
cv2.destroyAllWindows()"""


output_folder = './finalcode/input'
if not os.path.exists(output_folder):
  os.makedirs(output_folder)

# Save the frames to the "input" folder

video_name = os.path.basename(vid_path).split('.')[0]

for i, frame in frames.items():
  frame_filename = os.path.join(output_folder, f'{i}.jpg')
  cv2.imwrite(frame_filename, frame)
  

frames=enhance_photos(frames)
count=0
for i, frame in frames.items():
  frame_filename = os.path.join(output_folder, f'{video_name}_frame_00{count}.jpg')
  cv2.imwrite(frame_filename, frame)
  count+=1