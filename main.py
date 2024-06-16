import cv2

from detectFin import Detector
import mediapipe as mp

# def main():
#     cap = cv2.VideoCapture(0)
#     #detector = Detector()
#
#     mpHands = mp.solutions.hands
#     hands = mpHands.Hands()
#     mp_drawing = mp.solutions.drawing_utils
#
#     while True:
#         success, img = cap.read()
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         print('before',img)
#         results = hands.process(img)
#
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                      mp_drawing.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
#
#
#         print("img",img)
#         cv2.imshow('Finger Detection', img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#              break
#
#     cap.release()
#     cv2.destroyAllWindows()


def main():
    cap = cv2.VideoCapture(0)
    detector = Detector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.det_hands(frame)
        frame = detector.draw(frame)
        fingers = detector.count_fingers()


        cv2.putText(frame, f'Count: {fingers}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        cv2.imshow('Finger Detection', frame)

        # Выход при нажатии клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Hello World!")
    main()

