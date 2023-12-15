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

    def initiate_data_transformation(self, data_path):
        try:
            data = pd.read_csv(data_path)
            logging.info("Read raw data completed")
            logging.info("Initiating preprocessing on data")
            data = data.drop('img_name', axis=1)
            data['age'] = data['age'].apply(self.group)
            target_columns = ['gender', 'ethnicity','age']
            ### For complete data
            y = data[['gender', 'age']]
            X = data.drop(target_columns, axis=1)
            X = pd.Series(X['pixels'])
            X = X.apply(lambda x: x.split(' '))
            X = X.apply(lambda x: np.array(list(map(lambda z: np.int64(z), x))))
            X = np.array(X)
            X = np.stack(np.array(X), axis=0)
            X = np.reshape(X, (-1, 48, 48))
            y_gender = np.array(y['gender'])
            y_age = np.array(y['age'])
            X_gender_train, X_gender_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, train_size = 0.8)
            X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(X, y_age, train_size=0.8)
            logging.info("preprocessing of data done")
            
            Data = {
                'gender_data' : [X_gender_train, X_gender_test, y_gender_train, y_gender_test],
                'age_data' : [X_age_train, X_age_test, y_age_train, y_age_test]
            }

            return Data       
        except Exception as e:
            logging.info("Exception raised: {}".format(e))
            raise CustomException(e, sys)

    def group(self, age) -> int:
        if age > 0 and age < 9:
            return 0
        elif age > 9 and age < 15:
            return 1
        elif age > 15 and age < 24:
            return 2
        elif age > 24 and age < 40:
            return 3
        elif age > 40 and age < 60:
            return 4
        elif age > 60 and age < 85:
            return 5
        else:
            return 6