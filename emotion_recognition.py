import cv2
import numpy as np
import tensorflow as tf
import os


def preprocess_face(face_img: np.ndarray):
    face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img_resized = cv2.resize(face_img_gray, (128, 128))
    cv2.imwrite('tmp.jpg', face_img_resized)
    img = cv2.imread('tmp.jpg') / 255.0
    img = img.astype(np.float32)
    face_extra_dim = tf.expand_dims(img, 0)
    os.remove('tmp.jpg')
    return face_extra_dim


def predict_emotion(face_img: np.ndarray):
    labels = {
        0: 'angry',
        1: 'disgust',
        2: 'sad',
        3: 'happy',
        4: 'neutral',
        5: 'sad',
        6: 'surprise',
    }

    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], face_img)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return labels[np.argmax(output_data)]
