import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

class DataTransformation:
    def __init__(self):
        pass

    def initiate_data_transformation(self, image_path, label_path):
        try:
            label_data = pd.read_csv(label_path)
            logging.info("Read labels data completed")
            logging.info("Initiating preprocessing on label data")
            # data['age'] = data['age'].apply(self.group)
            images_data = np.load(image_path)
            ### For complete data
            logging.info("preprocessing of data done")
            y_gender = np.array(label_data['Genders'])
            y_age = np.array(label_data['Ages'])
            X_gender_train, X_gender_test, y_gender_train, y_gender_test = train_test_split(images_data, y_gender, train_size = 0.8, stratify=y_gender)
            X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(images_data, y_age, train_size=0.8, stratify=y_age)
            logging.info("Data splitted into train and test for training and validation.")
            Data = {
                'gender_data' : [X_gender_train, X_gender_test, y_gender_train, y_gender_test],
                'age_data' : [X_age_train, X_age_test, y_age_train, y_age_test]
            }
            return Data       
        except Exception as e:
            logging.info("Exception raised: {}".format(e))
            raise CustomException(e, sys)