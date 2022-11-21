import cv2
from face_detector import FaceDetector

from emotion_recognition import preprocess_face, predict_emotion

vid = cv2.VideoCapture(0)

while True:

    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)

    detector = FaceDetector(frame)
    success, faces = detector.detect_face()

    if not success:
        txt = cv2.putText(frame, "Can't detect faces", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                          cv2.LINE_AA)
        cv2.imshow('frame', frame)
    else:
        for i, face in enumerate(faces):
            x, y, w, h = face
            face_cropped = frame[y:y + h, x:x + w]
            face_for_detection = preprocess_face(face_cropped)
            prediction = predict_emotion(face_for_detection)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            image = cv2.putText(frame, prediction, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
            cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
