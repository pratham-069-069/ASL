# %%

# Import necessary libraries
import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
import keyboard
from tensorflow.keras.models import load_model
import language_tool_python

# Set the path to the data directory
PATH = os.path.join('data')

# Create an array of action labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Load the trained model
model = load_model('my_model.keras')

# Create an instance of the grammar correction tool
# tool = language_tool_python.LanguageToolPublicAPI('en-UK')

# Initialize the lists
sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

# Detection box parameters (normalized coordinates 0-1)
BOX_LEFT = 0.25      # 25% from left
BOX_RIGHT = 0.75     # 75% from left
BOX_TOP = 0.15       # 15% from top
BOX_BOTTOM = 0.85    # 85% from top

# Stability check parameters
PREDICTION_BUFFER_SIZE = 3  # Number of consistent predictions needed
CONFIDENCE_THRESHOLD = 0.92  # Increased from 0.9 for better accuracy
prediction_buffer = []

# Cooldown to prevent rapid re-detection of same letter
COOLDOWN_FRAMES = 15  # Number of frames to wait before detecting same letter again
cooldown_counter = 0
last_detected_action = None

def is_hand_in_box(results, box_left, box_right, box_top, box_bottom):
    """
    Check if either hand is within the detection box.
    Returns True if at least one hand has majority of landmarks in the box.
    """
    hands_in_box = []
    
    # Check left hand
    if results.left_hand_landmarks:
        landmarks = results.left_hand_landmarks.landmark
        in_box_count = sum(1 for lm in landmarks 
                          if box_left <= lm.x <= box_right and box_top <= lm.y <= box_bottom)
        # Require at least 15 out of 21 landmarks to be in box
        hands_in_box.append(in_box_count >= 15)
    
    # Check right hand
    if results.right_hand_landmarks:
        landmarks = results.right_hand_landmarks.landmark
        in_box_count = sum(1 for lm in landmarks 
                          if box_left <= lm.x <= box_right and box_top <= lm.y <= box_bottom)
        hands_in_box.append(in_box_count >= 15)
    
    # Return True if at least one hand is in box
    return any(hands_in_box) if hands_in_box else False

def draw_detection_box(image, box_left, box_right, box_top, box_bottom, hand_in_box=False):
    """
    Draw the detection box on the image.
    Color changes based on whether hand is detected inside.
    """
    h, w, _ = image.shape
    
    # Convert normalized coordinates to pixel coordinates
    left = int(box_left * w)
    right = int(box_right * w)
    top = int(box_top * h)
    bottom = int(box_bottom * h)
    
    # Choose color: green if hand in box, red otherwise
    color = (0, 255, 0) if hand_in_box else (0, 0, 255)
    thickness = 3 if hand_in_box else 2
    
    # Draw rectangle
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
    
    # Add text instruction
    text = "DETECTION ZONE" if hand_in_box else "Place hand in box"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = (left + right - text_size[0]) // 2
    text_y = top - 10
    
    cv2.putText(image, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def add_to_prediction_buffer(predicted_action, confidence):
    """
    Add prediction to buffer and check for stability.
    Returns the action if it's stable, None otherwise.
    """
    global prediction_buffer
    
    # Add to buffer
    prediction_buffer.append((predicted_action, confidence))
    
    # Keep buffer at specified size
    if len(prediction_buffer) > PREDICTION_BUFFER_SIZE:
        prediction_buffer.pop(0)
    
    # Check if buffer is full
    if len(prediction_buffer) < PREDICTION_BUFFER_SIZE:
        return None
    
    # Check if all predictions in buffer are the same
    actions_in_buffer = [p[0] for p in prediction_buffer]
    if len(set(actions_in_buffer)) == 1:  # All predictions are same
        # Check if average confidence is above threshold
        avg_confidence = np.mean([p[1] for p in prediction_buffer])
        if avg_confidence >= CONFIDENCE_THRESHOLD:
            return actions_in_buffer[0]
    
    return None

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a holistic object for sign prediction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Run the loop while the camera is open
    while cap.isOpened():
        # Read a frame from the camera
        _, image = cap.read()
        # Process the image and obtain sign landmarks using image_process function from my_functions.py
        results = image_process(image, holistic)
        
        # Check if hand is in detection box
        hand_in_box = is_hand_in_box(results, BOX_LEFT, BOX_RIGHT, BOX_TOP, BOX_BOTTOM)
        
        # Draw the detection box on the image
        draw_detection_box(image, BOX_LEFT, BOX_RIGHT, BOX_TOP, BOX_BOTTOM, hand_in_box)
        
        # Draw the sign landmarks on the image using draw_landmarks function from my_functions.py
        draw_landmarks(image, results)
        
        # Update cooldown counter
        if cooldown_counter > 0:
            cooldown_counter -= 1
        
        # Only collect keypoints if hand is in box
        if hand_in_box:
            # Extract keypoints from the pose landmarks
            keypoints.append(keypoint_extraction(results))

            # Check if 10 frames have been accumulated
            if len(keypoints) == 10:
                # Convert keypoints list to a numpy array
                keypoints_array = np.array(keypoints)
                # Make a prediction on the keypoints using the loaded model
                prediction = model.predict(keypoints_array[np.newaxis, :, :], verbose=0)
                # Clear the keypoints list for the next set of frames
                keypoints = []

                # Get predicted action and confidence
                predicted_action = actions[np.argmax(prediction)]
                confidence = np.amax(prediction)
                
                # Display current prediction confidence (for debugging)
                cv2.putText(image, f'Confidence: {confidence:.2f}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Add to stability buffer
                stable_action = add_to_prediction_buffer(predicted_action, confidence)
                
                # If we have a stable prediction and cooldown is over
                if stable_action and cooldown_counter == 0:
                    # Check if it's different from last prediction
                    if last_prediction != stable_action:
                        # Append the predicted sign to the sentence list
                        sentence.append(stable_action)
                        # Record new prediction
                        last_prediction = stable_action
                        last_detected_action = stable_action
                        # Start cooldown
                        cooldown_counter = COOLDOWN_FRAMES
                        # Clear buffer to start fresh
                        prediction_buffer = []
        else:
            # Clear keypoints if hand moves out of box
            if keypoints:
                keypoints = []
                prediction_buffer = []
            
            # Display instruction
            cv2.putText(image, 'Place hand in detection zone', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Limit the sentence length to 7 elements to make sure it fits on the screen
        if len(sentence) > 7:
            sentence = sentence[-7:]

        # Reset if the "Spacebar" is pressed
        if keyboard.is_pressed(' '):
            sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []
            prediction_buffer = []
            cooldown_counter = 0
            last_detected_action = None

        # Check if the list is not empty
        if sentence:
            # Capitalize the first word of the sentence
            sentence[0] = sentence[0].capitalize()

        # Check if the sentence has at least two elements
        if len(sentence) >= 2:
            # Check if the last element of the sentence belongs to the alphabet (lower or upper cases)
            if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
                # Check if the second last element of sentence belongs to the alphabet or is a new word
                if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                    # Combine last two elements
                    sentence[-1] = sentence[-2] + sentence[-1]
                    sentence.pop(len(sentence) - 2)
                    sentence[-1] = sentence[-1].capitalize()

        # Perform grammar check if "Enter" is pressed
        if keyboard.is_pressed('enter'):
            # Record the words in the sentence list into a single string
            text = ' '.join(sentence)
            # Apply grammar correction tool and extract the corrected result
            # grammar_result = tool.correct(text)

        if grammar_result:
            # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
            textsize = cv2.getTextSize(grammar_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2

            # Draw the sentence on the image
            cv2.putText(image, grammar_result, (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
            textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2

            # Draw the sentence on the image
            cv2.putText(image, ' '.join(sentence), (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the image on the display
        cv2.imshow('Camera', image)

        cv2.waitKey(1)

        # Check if the 'Camera' window was closed and break the loop
        if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Shut off the server
    # tool.close()
