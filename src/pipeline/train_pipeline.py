from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from pathlib import Path
import sys


class TrainPipeline:
    """
    A class that defines the training pipeline.
    """

    # def __init__(self, config_path: Path, paths_path: Path):
    #     """
    #     Initializes the TrainPipeline object.
    #
    #     Args:
    #     - config_path (Path): The path to the configuration file.
    #     - paths_path (Path): The path to the paths file.
    #     """
    #     self.config_path = config_path
    #     self.paths_path = paths_path

    def initiate_train_pipeline(self):
        """
        Initiates the training pipeline.
        """
        try:
            print("Trainig Started")
            logging.info("Training Pipeline started")

            logging.info("Ingestion Started")
            ingest_obj = DataIngestion()
            train_data_path, test_data_path = ingest_obj.ingest_data()
            logging.info("Ingestion Completed")

            logging.info("Transformation Started")
            train_data_transform = DataTransformation()
            preprocess_train_path = train_data_transform.initiate_data_transformation(train_data_path, is_train=True)

            test_data_transform = DataTransformation()
            preprocess_test_path = test_data_transform.initiate_data_transformation(test_data_path, is_train=False)
            logging.info("Transformation Completed")

            logging.info("Model Training Started")
            mt = ModelTrainer()
            roc_auc_score, model_path = mt.initiate_model_trainer(preprocess_train_path, preprocess_test_path)

            logging.info(f"Best Model roc_auc_score :-  {roc_auc_score}")

            logging.info("Model Training Completed")
            print("Model Training Completed")
            logging.info("---------------------- Train Pipeline Finished --------------------------")

        except CustomException as e:
            logging.exception(e)
            raise e

        except FileNotFoundError as e:
            logging.exception(f"File not found: {e.filename}")
            raise CustomException(f"File not found: {e.filename}", sys)

        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    tp = TrainPipeline()
    tp.initiate_train_pipeline()
