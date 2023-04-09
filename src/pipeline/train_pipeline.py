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
            logging.info("Training Pipeline started")

            logging.info("Ingestion Started")
            ingest_obj = DataIngestion()
            train_data_path, test_data_path = ingest_obj.ingest_data()
            logging.info("Ingestion Completed")

            logging.info("Transformation Started")
            train_data_transform = DataTransformation()
            preprocess_train_path = train_data_transform.initiate_data_transformation(train_data_path, is_train=True)

            logging.info(" In the middle....")

            test_data_transform = DataTransformation()
            preprocess_test_path = test_data_transform.initiate_data_transformation(test_data_path, is_train=False)
            logging.info("Transformation Completed")

            logging.info("Model Training Started")
            mt = ModelTrainer()
            roc_auc_score, model_path = mt.initiate_model_trainer(preprocess_train_path, preprocess_test_path)

            logging.info(f"Best Model roc_auc_score :-  {roc_auc_score}")

            logging.info("Model Training Completed")
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
    #
    # config_path = Path(__file__).parent.parent.parent.resolve() / "cfg" / "config.py"
    # paths_path = Path(__file__).parent.parent.parent.resolve() / "cfg" / "paths.py"

    # print(config_path, paths_path)
    # tp = TrainPipeline(config_path, paths_path)
    tp = TrainPipeline()
    tp.initiate_train_pipeline()



# from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation
# from src.components.model_trainer import ModelTrainer
#
# from src.exception import CustomException
# from src.logger import logging
# import os
# from pathlib import Path
# import sys
#
#
# class TrainPipeline:
# 	def __init__(self, config, paths):
# 		self.config = config
# 		self.paths = paths
#
# 	def initiate_train_pipeline(self):
# 		try:
# 			logging.info("Training Pipeline started")
#
# 			print("Ingestion Started")
# 			ingest_obj = DataIngestion()
# 			train_data_path, test_data_path = ingest_obj.initiate_data_ingestion(self.config, self.paths)
# 			print("Ingestion Completed")
#
# 			print("Transformation Started")
# 			train_data_transform = DataTransformation(self.config, self.paths)
# 			preprocess_train_path = train_data_transform.initiate_data_transformation(train_data_path, is_train=True)
#
# 			print(" In the middle....")
#
# 			test_data_transform = DataTransformation(self.config, self.paths)
# 			preprocess_test_path = test_data_transform.initiate_data_transformation(test_data_path, is_train=False)
# 			print("Transformation Completed")
#
# 			print("Model Training Started")
# 			mt = ModelTrainer(self.config, self.paths)
# 			roc_auc_score, model_path = mt.initiate_model_trainer(preprocess_train_path, preprocess_test_path)
#
# 			logging.info(f"Best Model roc_auc_score :-  {roc_auc_score}")
#
# 			print("Model Training Completed")
# 			print("---------------------- Train Pipeline Finished --------------------------")
#
# 		except Exception as e:
# 			logging.exception(e)
# 			raise CustomException(e, sys)
#
# if __name__ == "__main__":
# 	cfg_dir = "cfg"
# 	config_fname = "config.yaml"
# 	paths_fname = "paths.yaml"
#
# 	root = str(Path(os.getcwd()).parents[1])
#
# 	conf_ = os.path.join(root, cfg_dir, config_fname)
# 	paths_ =  os.path.join(root, cfg_dir, paths_fname)
#
# 	tp = TrainPipeline(conf_, paths_)
# 	tp.initiate_train_pipeline()
#
#
#


