import os
import sys

import pandas as pd
from pathlib import Path


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models
from cfg.config import Config
from cfg.paths import Paths

class ModelTrainer:
	def __init__(self):
		self.config = Config()
		self.paths = Paths()

	def split_the_data_X_y(self, df):
		X = df.drop(columns=['fraud_reported_Y'])
		y = df['fraud_reported_Y']
		return X, y

	def initiate_model_trainer(self, preprocessed_train_pth, preprocessed_test_pth):
		try:
			logging.info('model trainer initiated')
			logging.info('Split train and test data into X y initiated')

			train_df = pd.read_csv(preprocessed_train_pth)
			test_df = pd.read_csv(preprocessed_test_pth)

			X_train, y_train = self.split_the_data_X_y(train_df)
			X_test, y_test = self.split_the_data_X_y(test_df)

			logging.info('Split train and test data into X y completed')

			models = {
				# 'Balanced RFC' : BalancedRandomForestClassifier(),
				'Balanced Bagging Classifer': BalancedBaggingClassifier(estimator=DecisionTreeClassifier()),
				'Easy Ensemble Classifer': EasyEnsembleClassifier(),

			}

			params = {
				'Balanced RFC': {
					"random_state": [42],
					"n_estimators": [100, 200, 300, 400],
					"max_depth": [1, 2, 3, 4],

				},
				'Balanced Bagging Classifer': {
					"n_estimators": [150],
					"sampling_strategy": [0.8, 0.85],
					"replacement": [ False]
				},
				'Easy Ensemble Classifer': {
					"random_state": [34],
					"n_estimators": [200],
					"sampling_strategy": [0.75],
					"replacement": [False]

				}
			}
			cv = self.config.model_trainer.cv
			model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params, cv=cv)

			best_model_score = max(sorted(model_report.values()))

			best_model_name = list(model_report.keys())[
				list(model_report.values()).index(best_model_score)
			]
			best_model = models[best_model_name]

			if best_model_score < self.config.model_trainer.best_model_score_threshold:
				raise CustomException('No best model found', "")
			logging.info(f'Best model {best_model} found on both training and testing dataset')

			artifacts_dir = self.paths.artifacts_dir
			trained_model_fname = self.paths.model_trainer.trained_model_fname
			trained_model_path = Path("..") / ".." / artifacts_dir / trained_model_fname

			save_object(file_path=trained_model_path, obj=best_model)
			predicted = best_model.predict(X_test)
			roc_auc = roc_auc_score(y_test, predicted)
			return roc_auc, trained_model_path

		except Exception as e:
			logging.exception(e)
			raise CustomException(e, sys)
