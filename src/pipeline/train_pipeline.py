import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
import sys
from src.exception import CustomException


class TrainPipeline:
    def __init__():
        pass

    def train():
        try:
            obj = DataIngestion()
            image_data_path, labels_path = obj.initiate_data_ingestion()
            data_transformation = DataTransformation()
            data = data_transformation.initiate_data_transformation(image_data_path, labels_path)
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_trainer(data)
            logging.info("Training completed.")
        except Exception as e:
            logging.info("Exception Occured: {}".format(e))
            CustomException(e, sys)