from flask import Flask, render_template, redirect
import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import win32com.client
import time

app = Flask(__name__)

model_dict = pickle.load(open('./model_r.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z'}

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/video')
def video():
    global predicted_word
    cap = cv2.VideoCapture(0)
    predicted_word = ""
    appending = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        hand_present = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            data_aux = []
            x_ = []
            y_ = []
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
            if len(data_aux) > 0:
                hand_present = True
                data_aux = np.asarray(data_aux).reshape(1, -1)
                prediction = model.predict(data_aux)
                prediction_proba = model.predict_proba(data_aux)
                confidence_score = np.max(prediction_proba)

                if confidence_score >= .95:
                    predicted_character = labels_dict.get(int(prediction[0]), "Key not found")

                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10

                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, f"{predicted_character} ({confidence_score:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                    appending.append(predicted_character)
                    predicted_word = ''.join(appending)

                    # Slow down the appending by adding a delay
                    time.sleep(1.2)  # Adjust the delay time as needed

                if not hand_present:
                    appending.append(' ')

        predicted_word_length = len(predicted_word)
        if predicted_word_length > 0:
            font_scale = min(2, (W / (predicted_word_length * 30)))  # Adjust the constant multiplier as needed
        else:
            font_scale = 1  # Default font scale if predicted word is empty

        thickness = max(1, int(font_scale))

        font = cv2.FONT_HERSHEY_SIMPLEX

        text_size = cv2.getTextSize(predicted_word, font, font_scale, thickness)[0]

        text_x = int((W - text_size[0]) / 2)
        text_y = int((H + text_size[1]) / 2)

        cv2.putText(frame, predicted_word, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e') and appending:  # Press 'e' to erase the most recently appended alphabet
            appending.pop()  # Remove the last element from the appending list
            predicted_word = ''.join(appending)

    cap.release()
    cv2.destroyAllWindows()

    speaker = win32com.client.Dispatch("SAPI.SpVoice")

    return render_template('word.html', predicted_word=predicted_word)

@app.route('/play', methods=['POST'])
def play():
    global predicted_word
    if predicted_word:
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        voice = speaker.Speak(predicted_word)
        return render_template('word.html', predicted_word=predicted_word)

@app.route('/homee')
def homee():
    return redirect('home')

if __name__ == "__main__":
    app.run(debug=True)
