from src.pipeline.train_pipeline import TrainPipeline
import sys
from src.exception import CustomException
from src.logger import logging

if __name__ == '__main__':
    try:
        trainig_obj = TrainPipeline()
        trainig_obj.train()
    except Exception as e:
        logging.info("Exception Occured: {}".format(e))
        raise CustomException(e, sys)