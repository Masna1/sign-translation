import pickle
import cv2
import mediapipe as mp
import numpy as np

def playyvideo(model1):
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    labels_dict_words = {0:'airplane ', 1:'telephone', 2:'mother'}  # Update label dictionary with correct classes and characters
    predicted_word = ""
    appending = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
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
                data_aux = np.asarray(data_aux).reshape(1, -1)
                prediction = model1.predict(data_aux)
                predicted_character = labels_dict_words.get(int(prediction[0]), "Key not found")
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,cv2.LINE_AA)
                if cv2.waitKey(1) == ord('c'):
                    appending.append(predicted_character)
                    predicted_word = ''.join(appending)
        cv2.putText(frame, predicted_word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Predicted Word:", predicted_word)
    return predicted_word

# Load the trained model
model_dict_words = pickle.load(open('./model.p', 'rb'))
model1 = model_dict_words['model1']

# Call the playvideo function with the loaded model
predicted_word = playyvideo(model1)
