import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_model
from src.logger import logging
import mtcnn
import matplotlib.pyplot as plt
from  PIL import Image
import numpy as np

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, image):
        try:
            gender_model_path = 'artifacts\gender_model.json'
            age_model_path = 'artifacts\\age_model.json'
            gender_model = load_model(gender_model_path)
            age_model = load_model(age_model_path)
            preprocessed_data = self.preprocess(image)
            gender = np.round(gender_model.predict(preprocessed_data))
            age = age_model.predict(preprocessed_data)
            if gender == 0:
                gender = 'male'
            elif gender == 1:
                gender == 'female'
            results = {
                'gender' : gender,
                'age' : age
            }
            print(results)
        except Exception as e:
            raise CustomException(e, sys)
    
    def preprocess(self, image):
        try:
            detected_face = self.detect_face(image)
            extracted_face = self.extract_face(image, detected_face)
            data = np.asarray(extracted_face)
            data = data.astype('float32')
            data /= 255.0
            data = data.reshape((-1, 200, 200, 3))
            print(data.shape)
            return data
        except Exception as e:
            CustomException(e ,sys)

    def extract_face(self, filename, face_list):
        data = Image.open(filename)
        for i in range(len(face_list)):
            x1, y1, width, height = face_list[i]["box"]
            x2, y2 = x1 + width, y1 + height
            img = data.crop((x1, y1, x2, y2))
            resized_img = img.resize((200,200), Image.LANCZOS)
            # img = np.asarray(resized_img)
            # print(img.shape)
        return resized_img

    def detect_face(self, image):
        pixels = plt.imread(image)
        detector = mtcnn.MTCNN()
        faces = detector.detect_faces(pixels)
        return faces
    
if __name__ == '__main__':
    image_path = 'src\pipeline\IMG20220727195325.jpg'
    pipeline = PredictPipeline()
    pipeline.predict(image_path)