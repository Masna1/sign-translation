import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing data
DATA_DIR = './data1'

# Lists to store data and labels
data = []
labels = []

# Process images and extract hand landmarks
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []
        
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            data.append(data_aux)
            labels.append(dir_)

# Save processed data and labels to a pickle file
with open('data2.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Flatten and reshape data to have consistent lengths
max_sequence_length = max(len(seq) for seq in data)
flattened_data = [item for sublist in data for item in sublist]
reshaped_data = [flattened_data[i:i+max_sequence_length] for i in range(0, len(flattened_data), max_sequence_length)]

# Normalize coordinates to a fixed range [0, 1]
x_min = min(flattened_data)
x_max = max(flattened_data)
normalized_data = [[(x - x_min) / (x_max - x_min) for x in sequence] for sequence in reshaped_data]

# Check for class imbalance and address if necessary
label_distribution = Counter(labels)
print("Label distribution:", label_distribution)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(normalized_data, labels, test_size=0.2, random_state=42)

# Add comments to improve code readability
# Normalize coordinates and reshape data
# Check label distribution for class imbalance
# Split data into training and validation sets

# Explore different configurations of MediaPipe
# Experiment with different machine learning models
