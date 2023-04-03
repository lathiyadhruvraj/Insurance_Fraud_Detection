import sys
from dataclasses import dataclass
from pickle import dump, load

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
	std_scaler_pth = r'D:\projects\Insurance_Fraud_Detection\artifacts\std_scaler.pkl'

	imputer_pth = r'D:\projects\Insurance_Fraud_Detection\artifacts\knn_imputer.pkl'

	preprocessed_train_pth = r'D:\projects\Insurance_Fraud_Detection\artifacts\train_preprocessed.csv'
	preprocessed_test_pth = r'D:\projects\Insurance_Fraud_Detection\artifacts\test_preprocessed.csv'
	preprocessed_pred_pth = r'D:\projects\Insurance_Fraud_Detection\artifacts\prediction_preprocessed.csv'

	knn_n_nbrs = 3

	cols_to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location',
							'incident_date','incident_state','incident_city','insured_hobbies','auto_make',
							'auto_model','auto_year', 'age', 'total_claim_amount']

	custom_mapping_columns = ['policy_csl', 'insured_education_level', 'incident_severity', 'insured_sex',
	                          'property_damage', 'police_report_available', 'fraud_reported']

	dummies_columns = ['insured_occupation', 'insured_relationship', 'incident_type', 'collision_type',
                   'authorities_contacted']

	numerical_columns = ['months_as_customer', 'policy_deductable', 'umbrella_limit',
	                     'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
	                     'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim',
	                     'property_claim',
	                     'vehicle_claim']

class DataTransformation:
	def __init__(self):
		self.data_transformation_config = DataTransformationConfig()

	def drop_unnecessary_cols(self, df):
		"""
				This function is  for removing unnecessary columns
		"""
		try:

			df.drop(columns=self.data_transformation_config.cols_to_drop, inplace=True)

			return df

		except Exception as e:
			logging.exception(e)
			raise CustomException(e, sys)

	def replace_unknown_value_with_nan(self, df):
		"""
				This function is  for replacing unknown values with NaN

		"""
		try:
			unknown_to_replace = ['?']

			for uk in unknown_to_replace:
				df = df.replace(uk, np.nan)

			return df

		except Exception as e:
			logging.exception(e)
			raise CustomException(e, sys)

	def custom_mapping_for_encoding(self, df):
		"""
				This function is  for custom encoding

		"""
		try:

			# custom mapping for encoding
			df['policy_csl'] = df['policy_csl'].map({'100/300': 1, '250/500': 2.5, '500/1000': 5})
			df['insured_education_level'] = df['insured_education_level'].map(
				{'JD': 1, 'High School': 2, 'College': 3, 'Masters': 4, 'Associate': 5, 'MD': 6, 'PhD': 7})
			df['incident_severity'] = df['incident_severity'].map(
				{'Trivial Damage': 1, 'Minor Damage': 2, 'Major Damage': 3, 'Total Loss': 4})
			df['insured_sex'] = df['insured_sex'].map({'FEMALE': 0, 'MALE': 1})
			df['property_damage'] = df['property_damage'].map({'NO': 0, 'YES': 1})
			df['police_report_available'] = df['police_report_available'].map({'NO': 0, 'YES': 1})
			df['fraud_reported'] = df['fraud_reported'].map({'N': 0, 'Y': 1})

			return df

		except Exception as e:
			logging.exception(e)
			raise CustomException(e, sys)

	def get_dummies_for_cat_val(self, df):
		"""
				This function is  for custom encoding

		"""
		try:

			# custom mapping for encoding
			cat_df = df.select_dtypes(include=['object']).copy()

			for col in self.data_transformation_config.dummies_columns:
				cat_df = pd.get_dummies(cat_df, columns=[col], prefix=[col], drop_first=True)

			df.drop(columns=self.data_transformation_config.dummies_columns, inplace=True)

			cat_df = pd.concat([df, cat_df], axis=1)
			return cat_df

		except Exception as e:
			logging.exception(e)
			raise CustomException(e, sys)


	def apply_standard_scaler(self, df, is_train=False):
		"""
		:param df:  Standard Scaling the input df
		:param is_train: If True fit transform the train file , If false - transform the test/predict file
		:return: scaled df
		"""
		try:

			scaler = StandardScaler()
			num_df = df[self.data_transformation_config.numerical_columns]
			# num_df = df.select_dtypes(exclude=['object']).copy()
			cols = num_df.columns
			if is_train:
				scaled_df = scaler.fit_transform(num_df)
				dump(scaler, open(self.data_transformation_config.std_scaler_pth, 'wb'))
				logging.info("standard scaler object dumped after fit_transform")
			else:
				scaler = load(open(self.data_transformation_config.std_scaler_pth, 'rb'))
				scaled_df = scaler.transform(num_df)
				logging.info("standard scaler object loaded for transform df")

			scaled_df = pd.DataFrame(scaled_df, columns=cols)

			return scaled_df

		except Exception as e:
			logging.exception(e)
			raise CustomException(e, sys)

	def apply_KNN_imputer(self, final_df, is_train=False):
		"""
		:param final_df: KNN imputation for missing data in categorical column
		:param is_train: If True fit transform the train file , If false - transform the test/predict file
		:return: imputed_df
		"""
		try:
			imputer = KNNImputer(n_neighbors=self.data_transformation_config.knn_n_nbrs, missing_values=np.nan)

			cols = final_df.columns
			if is_train:
				imputed_df = imputer.fit_transform(final_df)
				dump(imputer, open(self.data_transformation_config.imputer_pth, 'wb'))
				logging.info("KNN Imputer object dumped after fit_transform")
			else:
				imputer = load(open(self.data_transformation_config.imputer_pth, 'rb'))
				imputed_df = imputer.transform(final_df)
				logging.info("KNN Imputer object loaded for transform df")

			imputed_df = pd.DataFrame(imputed_df, columns=cols)

			return  imputed_df

		except Exception as e:
			logging.exception(e)
			raise CustomException(e, sys)

	def initiate_data_transformation(self, file_path, is_train=False, is_pred=False):
		"""
		:param file_path: handles methods of Data Tansformation Class
		:param is_train: If True fit transform the train file , If false - transform the test/predict file
		:param is_pred: for identifying between test file and predict file
		:return: path to preprocessed file
		"""
		try:
			if is_train:
				logging.info("TRANSFORMATION FOR TRAIN DATA INITIATED")
			else:
				logging.info("TRANSFORMATION FOR TEST/PRED DATA INITIATED")

			logging.info("==============================================")

			df = pd.read_csv(file_path)
			logging.info("Read of data completed")
			logging.info(f"original  shape - {df.shape}")

			df = self.drop_unnecessary_cols(df)
			logging.info("Deleted unnecessary columns")
			logging.info(f"after drop_unnecessary_cols {df.shape}" )

			df = self.replace_unknown_value_with_nan(df)
			logging.info("Unknown values replaced with NaN")
			logging.info(f"after replace_unknown_value_with_nan {df.shape}")

			df = self.custom_mapping_for_encoding(df)
			logging.info("Custom mapping completed")
			logging.info(f"after custom_mapping_for_encoding {df.shape}")

			cat_df = self.get_dummies_for_cat_val(df)
			logging.info("Dummies Added")
			logging.info(f"after get_dummies_for_cat_val {cat_df.shape}")

			scaled_df = self.apply_standard_scaler(df, is_train)
			logging.info("Standard Scaler Completed")
			logging.info(f"after apply_standard_scaler {scaled_df.shape}")

			final_df = pd.concat([scaled_df, cat_df], axis=1)
			logging.info(f"Final df shape {final_df.shape}")

			imputed_df = self.apply_KNN_imputer(final_df, is_train)
			logging.info("KNN imputation Completed")
			logging.info(f"after apply_KNN_imputer {imputed_df.shape}")

			if is_train:
				imputed_df.to_csv(self.data_transformation_config.preprocessed_train_pth)
				logging.info("TRANSFORMATION FOR TRAIN DATA COMPLETED")
				logging.info("=======================================")

				return self.data_transformation_config.preprocessed_train_pth

			elif is_pred:
				imputed_df.to_csv(self.data_transformation_config.preprocessed_pred_pth)
				logging.info("TRANSFORMATION FOR PRED DATA COMPLETED")
				logging.info("=======================================")
				return self.data_transformation_config.preprocessed_pred_pth

			else:
				imputed_df.to_csv(self.data_transformation_config.preprocessed_test_pth)
				logging.info("TRANSFORMATION FOR TEST DATA COMPLETED")
				logging.info("=======================================")

				return self.data_transformation_config.preprocessed_test_pth

		except Exception as e:
			logging.exception(e)
			raise CustomException(e, sys)
