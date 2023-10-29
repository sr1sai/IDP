import cv2
import numpy as np
import os
import time

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
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frames.append(prev_frame)

    while True:
        ret, frame = cap.read()

        # Break the loop when we reach the end of the video
        if not ret:
            break

        # Convert the frame to grayscale for comparison
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the absolute difference between the current and previous frames
        frame_diff = cv2.absdiff(prev_frame_gray, frame_gray)

        # If the frame is not a duplicate (based on a simple threshold),
        # add it to the list of frames and update the previous frame
        if np.mean(frame_diff) > 20:
            frames.append(frame)
            prev_frame_gray = frame_gray.copy()
            frame_count+=1
            
        # Display the frame (optional)
        #cv2.imshow('Frame', frame)
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Release the video capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()
    print("no of frames: ",frame_count)

    video_name = os.path.basename(vid_path).split('.')[0]

    # frames_dict= { frame_name:frame 

    frames_dict={}
    
    for i, frame in enumerate(frames):
        frames_dict[video_name+"_frame_"+str(i)]=frame
    
    return frames_dict


vid_path="./phase_1/vid2.mp4"
frames=extractor(vid_path)
print(frames.keys())
for frame_name, frame in frames.items():
    cv2.imshow(frame_name, frame)
    cv2.waitKey(0)  # Display each frame until any key is pressed
    cv2.destroyWindow(frame_name)  # Close the frame window after viewing

cv2.waitKey(0)  # Wait for any key to be pressed to close all frame windows
cv2.destroyAllWindows()