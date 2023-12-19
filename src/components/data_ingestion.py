import os
import sys
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from PIL import Image
import numpy as np

class DataIngestionConfig:
    images_data_path: str=os.path.join('artifacts',"image_data.npy")
    gender_age_data_path: str=os.path.join('artifacts', "age_gender.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # images = []
            # ages = []
            # genders = []
            # logging.info("Preparing data for images reading, conversion and their respective age and gender.")
            # file_path = 'notebook\data\image_data'
            # for i in os.listdir(file_path):
            #     split = i.split('_')
            #     ages.append(int(split[0]))
            #     genders.append(int(split[1]))
            #     img = Image.open(file_path + '\\' + i)
            #     img.resize((200,200), Image.LANCZOS)
            #     ar = np.asarray(img)
            #     images.append(ar)
            #     img.close()
            # image_arr = np.array(images)
            # os.makedirs(os.path.dirname(self.ingestion_config.images_data_path),exist_ok=True)
            # np.save(self.ingestion_config.images_data_path, image_arr)
            # logging.info("Saved image data.")
            # ages = pd.Series(list(ages), name = 'Ages')
            # genders = pd.Series(list(genders), name = 'Genders')
            # df = pd.concat([ages, genders], axis=1)
            # df.to_csv(DataIngestionConfig.gender_age_data_path, index=False, header=True)
            # logging.info("Saved age and gender labels")
            # logging.info("Ingestion of data is completed")
            return (
                self.ingestion_config.images_data_path,
                self.ingestion_config.gender_age_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    image_data_path, labels_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data = data_transformation.initiate_data_transformation(image_data_path, labels_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(data)