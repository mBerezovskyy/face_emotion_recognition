import cv2
import numpy as np


class FaceDetector:
    def __init__(self, image: np.ndarray):
        self.face_config_path = 'haarcascade_frontalface_default.xml'
        self.eye_config_path = 'haarcascade_eye.xml'
        self.image = image

    def __initialize_face_cascade(self):
        face_cascade = cv2.CascadeClassifier(self.face_config_path)
        if face_cascade.empty():
            raise 'Error loading face cascade'
        return face_cascade

    def __initialize_eye_cascade(self):
        eye_cascade = cv2.CascadeClassifier(self.eye_config_path)
        if eye_cascade.empty():
            raise 'Error loading eye cascade'
        return eye_cascade

    def detect_face(self):
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        face_cascade = self.__initialize_face_cascade()

        faces = face_cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.3,
            minNeighbors=3,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if faces == ():
            return False, ()

        face_coords = [face for face in faces]

        return True, face_coords
