import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from src.components.data_ingestion import DatatIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

## run data Ingestion

if __name__=='__main__':
    obj=DatatIngestion()
    train_data_path, test_data_path= obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr,_=data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr, test_arr)