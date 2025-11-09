# %%

# Import necessary libraries
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from itertools import product
from sklearn import metrics

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set the path to the data directory
PATH = os.path.join('data')

# Define the number of frames
frames = 10
min_sequences_required = 5  # Minimum sequences needed to include an action in training

# Function to count valid sequences for an action
def get_valid_sequences(action_path, frames):
    """Returns list of sequence numbers that have all required frames"""
    if not os.path.exists(action_path):
        return []
    
    valid_sequences = []
    sequences = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
    
    for seq in sequences:
        seq_path = os.path.join(action_path, seq)
        # Check if all frames exist
        all_frames_exist = all(
            os.path.exists(os.path.join(seq_path, f"{frame}.npy")) 
            for frame in range(frames)
        )
        if all_frames_exist:
            valid_sequences.append(int(seq))
    
    return sorted(valid_sequences)

# Get all potential actions from data directory
all_actions = [d for d in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, d))]

# Filter actions that have enough valid sequences
valid_actions = []
action_sequences = {}

print("\nScanning data directory...")
print("="*60)
for action in sorted(all_actions):
    action_path = os.path.join(PATH, action)
    valid_seqs = get_valid_sequences(action_path, frames)
    
    if len(valid_seqs) >= min_sequences_required:
        valid_actions.append(action)
        action_sequences[action] = valid_seqs
        print(f"Letter '{action.upper()}': {len(valid_seqs)} valid sequences - INCLUDED")
    else:
        print(f"Letter '{action.upper()}': {len(valid_seqs)} valid sequences - SKIPPED (need at least {min_sequences_required})")

print("="*60)

if not valid_actions:
    print("\nError: No actions have enough data for training!")
    print(f"Each action needs at least {min_sequences_required} complete sequences.")
    exit()

# Create array of valid actions and label map
actions = np.array(valid_actions)
label_map = {label:num for num, label in enumerate(actions)}

print(f"\nTraining model with {len(actions)} actions: {', '.join([a.upper() for a in actions])}")
print(f"Total sequences to load: {sum(len(seqs) for seqs in action_sequences.values())}")

# Initialize empty lists to store landmarks and labels
landmarks, labels = [], []

# Iterate over valid actions and their sequences to load landmarks and corresponding labels
print("\nLoading data...")
for action in actions:
    for sequence in action_sequences[action]:
        temp = []
        try:
            for frame in range(frames):
                npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
                temp.append(npy)
            landmarks.append(temp)
            labels.append(label_map[action])
        except Exception as e:
            print(f"Warning: Skipping {action}/sequence_{sequence} due to error: {e}")
            continue

print(f"Successfully loaded {len(landmarks)} sequences")

# Convert landmarks and labels to numpy arrays
X, Y = np.array(landmarks), to_categorical(labels).astype(int)

print(f"\nData shape: X={X.shape}, Y={Y.shape}")

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
print("\nBuilding model...")

# Define the model architecture
model = Sequential()
model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(10,126)))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(32, return_sequences=False, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print(f"Model compiled. Output classes: {actions.shape[0]}")
print("\nStarting training...")

# Train the model
model.fit(X_train, Y_train, epochs=100)

print("\nSaving model...")
# Save the trained model
model.save('my_model.keras')
print("Model saved as 'my_model.keras'")

print("\nEvaluating model...")
# Make predictions on the test set
predictions = np.argmax(model.predict(X_test), axis=1)
# Get the true labels from the test set
test_labels = np.argmax(Y_test, axis=1)

# Calculate the accuracy of the predictions
accuracy = metrics.accuracy_score(test_labels, predictions)

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Actions trained: {', '.join([a.upper() for a in actions])}")
print("="*60)
