import cv2
import mediapipe as mp


class Detector:
    def __init__(self):

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def det_hands(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(rgb)
        return image

    def draw(self, image):
        if self.result.multi_hand_landmarks:
            for hand_landmarks in self.result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return image

    def count_fingers(self):
        if not self.result.multi_hand_landmarks:
            return 0

        # hand_landmarks = self.result.multi_hand_landmarks
        # fingers_up = []
        total_fingers = 0
        for hand_landmarks in self.result.multi_hand_landmarks:
            fingers_up = []
            for lm_index in [4, 8, 12, 16, 20]:

                if lm_index == 4:
                    if hand_landmarks.landmark[4].y < hand_landmarks.landmark[5].y:
                        fingers_up.append(1)
                    else:
                        fingers_up.append(0)
                else:
                    if hand_landmarks.landmark[lm_index].y < hand_landmarks.landmark[lm_index - 2].y:
                        fingers_up.append(1)
                    else:
                        fingers_up.append(0)

            total_fingers += sum(fingers_up)
        return total_fingers
