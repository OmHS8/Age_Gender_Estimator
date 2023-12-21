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
            logging.info("Preprocessing the image.")
            preprocessed_data = self.preprocess(image)
            if (type(preprocessed_data) == bool and preprocessed_data == False):
                logging.info("Image contained no faces. Exited.")
                results = -1
            elif ((type(preprocessed_data) == bool) and preprocessed_data == True):
                results = 0
                logging.info("Image contained multiple faces")
            else:
                gender_model_path = 'artifacts\gender_model.json'
                age_model_path = 'artifacts\\age_model.json'
                logging.info("Initiating prediction process")
                logging.info("Loading gender model")
                gender_model = load_model(gender_model_path)
                logging.info("Loaded gender model")
                logging.info("Loading age model")
                age_model = load_model(age_model_path)
                logging.info("Loaded age model.")
                gender = np.round(gender_model.predict(preprocessed_data))
                age = age_model.predict(preprocessed_data)
                logging.info("Estimated age and gender for face in image")
                if gender == 0:
                    gender = 'male'
                elif gender == 1:
                    gender = 'female'
                age = int(np.round(age[0][0]))
                results = {
                    'gender' : gender,
                    'age' : age
                }
                logging.info("Age: {}\t\t Gender: {}".format(age, gender))
            return results
        except Exception as e:
            logging.info("Exception Occured: {}".format(e))
            raise CustomException(e, sys)
    
    def preprocess(self, image):
        try:
            logging.info("Detecting faces in image.")
            detected_face = self.detect_face(image)
            if (detected_face == -1):
                return False
            elif (detected_face == 0):
                return True
            logging.info("Extracting the face from image.")
            extracted_face = self.extract_face(image, detected_face)
            data = np.asarray(extracted_face)
            data = data.astype('float32')
            data /= 255.0
            data = data.reshape((-1, 200, 200, 3))
            logging.info("Image preprocessed successfully.")
            return data
        except Exception as e:
            logging.info("Exception Occured: {}".format(e))
            CustomException(e ,sys)

    def extract_face(self, filename, face_list):
        data = Image.open(filename)
        for i in range(len(face_list)):
            x1, y1, width, height = face_list[i]["box"]
            x2, y2 = x1 + width, y1 + height
            img = data.crop((x1, y1, x2, y2))
            resized_img = img.resize((200,200), Image.LANCZOS)
        return resized_img

    def detect_face(self, image):
        pixels = plt.imread(image)
        detector = mtcnn.MTCNN()
        faces = detector.detect_faces(pixels)  
        if (len(faces) == 0):
            return -1
        elif (len(faces) > 1):
            return 0
        return faces