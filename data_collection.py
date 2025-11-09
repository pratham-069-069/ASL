# %%

# Import necessary libraries
import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import *
import keyboard

# Define the actions (signs) that will be recorded and stored in the dataset
# Updated to include all 26 letters
actions = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])

# Define the number of sequences and frames to be recorded for each action
sequences = 30
frames = 10

# Set the path where the dataset will be stored
PATH = os.path.join('data')

# Function to count existing sequences for an action
def count_existing_sequences(action, path):
    action_path = os.path.join(path, action)
    if not os.path.exists(action_path):
        return 0
    existing = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
    return len(existing)

# Function to display available actions and their completion status
def display_actions_status(actions, path, total_sequences):
    print("\n" + "="*60)
    print("SIGN LANGUAGE DATA COLLECTION - Status")
    print("="*60)
    for i, action in enumerate(actions):
        completed = count_existing_sequences(action, path)
        status = f"[{completed}/{total_sequences}]"
        percentage = (completed / total_sequences) * 100
        print(f"{i+1:2d}. Letter '{action.upper()}': {status} ({percentage:.0f}% complete)")
    print("="*60)

# Function to get user's choice of action
def get_user_action_choice(actions, path, total_sequences):
    while True:
        display_actions_status(actions, path, total_sequences)
        print("\nEnter the number of the letter you want to record (or 'q' to quit):")
        choice = input("> ").strip().lower()
        
        if choice == 'q':
            return None
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(actions):
                return actions[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(actions)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")

# Get user's choice of which letter to record
selected_action = get_user_action_choice(actions, PATH, sequences)
if selected_action is None:
    print("Exiting...")
    exit()

# Count existing sequences for the selected action
existing_count = count_existing_sequences(selected_action, PATH)
print(f"\nSelected: '{selected_action.upper()}'")
print(f"Existing sequences: {existing_count}/{sequences}")
print(f"Will record sequences {existing_count} to {sequences-1}")

# Create directories for remaining sequences
for sequence in range(existing_count, sequences):
    try:
        os.makedirs(os.path.join(PATH, selected_action, str(sequence)))
    except:
        pass

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Variable to track if user wants to exit
should_exit = False

# Create a MediaPipe Holistic object for hand tracking and landmark extraction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Loop through remaining sequences and frames to record data
    for sequence in range(existing_count, sequences):
        if should_exit:
            break
            
        for frame in range(frames):
            if should_exit:
                break
                
            # If it is the first frame of a sequence, wait for the spacebar key press to start recording
            if frame == 0: 
                while True:
                    if keyboard.is_pressed('q'):
                        should_exit = True
                        break
                    if keyboard.is_pressed(' '):
                        break
                    _, image = cap.read()

                    results = image_process(image, holistic)
                    draw_landmarks(image, results)

                    # Calculate progress
                    progress = f"{sequence + 1}/{sequences}"
                    
                    cv2.putText(image, f'Recording data for "{selected_action.upper()}". Sequence {progress}.',
                                (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Paused.', (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'Press "Space" when ready or "Q" to quit.', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                    cv2.imshow('Camera', image)
                    cv2.waitKey(1)
                    
                    # Check if the 'Camera' window was closed and break the loop
                    if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
                        should_exit = True
                        break
                
                if should_exit:
                    break
            else:
                # For subsequent frames, directly read the image from the camera
                _, image = cap.read()
                # Process the image and extract hand landmarks using the MediaPipe Holistic pipeline
                results = image_process(image, holistic)
                # Draw the hand landmarks on the image
                draw_landmarks(image, results)

                # Calculate progress
                progress = f"{sequence + 1}/{sequences}"
                frame_progress = f"Frame {frame + 1}/{frames}"
                
                # Display text on the image indicating the action and sequence number being recorded
                cv2.putText(image, f'Recording data for "{selected_action.upper()}". Sequence {progress}. {frame_progress}',
                            (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                cv2.imshow('Camera', image)
                
                # Check if 'q' is pressed during recording
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    should_exit = True
                    break

            # Check if the 'Camera' window was closed and break the loop
            if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
                should_exit = True
                break

            # Extract the landmarks from both hands and save them in arrays
            keypoints = keypoint_extraction(results)
            frame_path = os.path.join(PATH, selected_action, str(sequence), str(frame))
            np.save(frame_path, keypoints)

    # Release the camera and close any remaining windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Display final status
    final_count = count_existing_sequences(selected_action, PATH)
    print(f"\nData collection completed!")
    print(f"Letter '{selected_action.upper()}': {final_count}/{sequences} sequences recorded")
    if final_count < sequences:
        print(f"Remaining: {sequences - final_count} sequences")
        print("Run the script again to continue recording.")
