import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

# change these if u record different gestures (order matters!!)
LABELS = ["open", "fist", "point", "peace"]


def get_features(hand, left=False):
    # turn mediapipe's 21 points into 63 numbers the model can use
    pts = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
    pts = pts - pts[0]  # move wrist to 0,0 so position on screen doesnt matter
    if left:
        pts[:, 0] = -pts[:, 0]  # mirror left hand so it looks like a right hand
    size = np.linalg.norm(pts[9, :2])  # wrist -> middle knuckle, so distance from cam doesnt matter
    if size > 0:
        pts = pts / size
    return pts.flatten()


class Net(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.layers(x)


class GestureThing:
    def __init__(self, path="gesture_model.pt"):
        self.net = Net(len(LABELS))
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()
        self.last_few = deque(maxlen=5)  # average last 5 frames so it stops flickering
        self.current = None
        self.just_changed = False

    def predict(self, hand, left=False):
        x = torch.tensor(get_features(hand, left)).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.net(x), dim=1)[0].numpy()

        self.last_few.append(probs)
        avg = np.mean(self.last_few, axis=0)
        i = int(avg.argmax())
        conf = float(avg[i])

        if conf < 0.7:  # not sure enough, ignore
            self.just_changed = False
            return None, conf

        label = LABELS[i]
        # only true on the frame the gesture switches, so stuff doesnt fire 60x a sec
        self.just_changed = label != self.current
        self.current = label
        return label, conf

    def reset(self):
        # call this when the hand leaves the screen
        self.last_few.clear()
        self.current = None
        self.just_changed = False


# test it with the webcam: python gesture.py
if __name__ == "__main__":
    import cv2
    import mediapipe as mp

    g = GestureThing()
    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    times = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        text = "no hand"
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            left = res.multi_handedness[0].classification[0].label == "Left"
            draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

            start = time.time()
            label, conf = g.predict(hand, left)
            times.append((time.time() - start) * 1000)

            text = f"{label} {conf:.2f}"
            if g.just_changed:
                print("->", label)
        else:
            g.reset()

        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("gestures", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if times:
        print(f"avg model time: {np.mean(times):.2f} ms")