import cv2
from detectFin import Detector


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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
