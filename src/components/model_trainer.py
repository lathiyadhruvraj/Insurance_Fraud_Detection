import os
import sys
from dataclasses import dataclass
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.ensemble import (
	AdaBoostClassifier,
	GradientBoostingClassifier,
	RandomForestClassifier,
)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
	trained_model_file_path = r'D:\projects\Insurance_Fraud_Detection\artifacts\model.pkl'
	best_model_score_threshold = 0.56

class ModelTrainer:
	def __init__(self):
		self.model_trainer_config = ModelTrainerConfig()

	def split_the_data_X_y(self, df):
		try:
			X = df.drop(columns=['fraud_reported'])
			y = df['fraud_reported']
			return X, y

		except Exception as e:
			logging.exception(e)
			raise CustomException(e, sys)

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
				# 'SVC' : SVC(),
				# 'Random Forest Classifier': RandomForestClassifier(),
				# 'Balanced RFC' : BalancedRandomForestClassifier(),
				'Balanced Bagging Classifer': BalancedBaggingClassifier(),
				'Easy Ensemble Classifer' : EasyEnsembleClassifier(),
				# 'Decision Tree Classifier': DecisionTreeClassifier(),
				# 'Gradient Boosting Classifier': GradientBoostingClassifier(),
				# 'XGBClassifier': XGBClassifier(),
				# 'CatBoosting Classifier': CatBoostClassifier(verbose=False),
				# 'AdaBoost Classifier': AdaBoostClassifier(),
			}
			params = {
				'Balanced RFC' : {
					"random_state": [42],
					"n_estimators": [100, 200, 300, 400],
					"max_depth": [1, 2, 3, 4],

				},
				'Balanced Bagging Classifer': {
					"estimator" : [DecisionTreeClassifier()],
			        "n_estimators" : [150],
			        "sampling_strategy" : [0.8,0.85],
			        "replacement" :[ False],
				},
				'Easy Ensemble Classifer': {
					"random_state": [34],
					"n_estimators": [200],
					"sampling_strategy": [0.75],
					"replacement": [ False]
				},
				'SVC':{
					'kernel':['rbf']
				},
				'Decision Tree Classifier': {
					'criterion': ['log_loss', 'entropy', 'gini'],
					'splitter':['best','random'],
					'max_depth' : [2, 3, 4],
				},
				'Random Forest Classifier': {
					'max_depth' : [2, 3, 4],
					'n_estimators': [ 8, 16, 32, 64]
				},
				'Gradient Boosting Classifier': {
					'subsample': [0.6, 0.65, 0.7, 0.75],
					'n_estimators': [16, 32, 64, 128],
					'max_depth': [2, 3, 4]
				},
				'XGBClassifier':{

					'max_depth': [2, 3]
				},
				'CatBoosting Classifier': {
					'learning_rate': [0.01, 0.05, 0.1],
					'iterations': [30, 50, 100],
					'max_depth': [2, 3, 4, 6, 8, 10]
				},
				'AdaBoost Classifier': {
					'learning_rate': [.1, .01, 0.5, .001],
					'n_estimators': [8, 16, 32, 64, 128]
				}

			}

			model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
			                                     models=models, param=params)

			## To get best model score from dict
			best_model_score = max(sorted(model_report.values()))

			## To get best model name from dict

			best_model_name = list(model_report.keys())[
				list(model_report.values()).index(best_model_score)
			]
			best_model = models[best_model_name]

			if best_model_score < self.model_trainer_config.best_model_score_threshold:
				raise CustomException( 'No best model found')
			logging.info(f'Best model {best_model} found on both training and testing dataset')

			save_object(
				file_path=self.model_trainer_config.trained_model_file_path,
				obj=best_model
			)

			predicted = best_model.predict(X_test)

			roc_auc = roc_auc_score(y_test, predicted)
			return roc_auc, self.model_trainer_config.trained_model_file_path

		except Exception as e:
			logging.exception(e)
			raise CustomException(e, sys)

if __name__ == '__main__' :
	trner = ModelTrainer()
	trner.initiate_model_trainer()