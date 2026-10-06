# record training data
# press 1-4 to start/stop recording that gesture, q to save + quit
# move ur hand around while recording (tilt it, closer/farther, both hands) or the model will be bad

import os

import cv2
import mediapipe as mp
import numpy as np

from gesture import LABELS, get_features

os.makedirs("data", exist_ok=True)

# keep old data if it exists so i can record more later
if os.path.exists("data/X.npy"):
    X = list(np.load("data/X.npy"))
    y = list(np.load("data/y.npy"))
else:
    X, y = [], []

hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
recording = None

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        left = res.multi_handedness[0].classification[0].label == "Left"
        draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        if recording is not None:
            X.append(get_features(hand, left))
            y.append(recording)

    if recording is None:
        cv2.putText(frame, "not recording", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    else:
        cv2.putText(frame, "REC " + LABELS[recording], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # show how many of each i have
    for i, name in enumerate(LABELS):
        count = y.count(i)
        cv2.putText(frame, f"{i+1}: {name} ({count})", (10, 60 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("collect", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    for i in range(len(LABELS)):
        if key == ord(str(i + 1)):
            recording = None if recording == i else i

cap.release()
cv2.destroyAllWindows()

np.save("data/X.npy", np.array(X, dtype=np.float32))
np.save("data/y.npy", np.array(y, dtype=np.int64))
print("saved", len(X), "samples")