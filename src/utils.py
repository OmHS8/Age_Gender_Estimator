import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from tensorflow.keras.models import model_from_json


def save_model(file_path, model):     
    try:
        model_json = model.to_json()
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w") as json_file:
            json_file.write(model_json)
        model.save_weights("{}_model.h5".format(file_path))

    except Exception as e:
        raise CustomException(e, sys)

def load_model(file_path):
    try:
        json_file = open(file_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("{}_model.h5".format(file_path))
        return loaded_model
    except Exception as e:
        raise CustomException(e, sys)

