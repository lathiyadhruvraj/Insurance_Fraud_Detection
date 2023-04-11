import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pickle import dump, load
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from cfg.config import Config
from cfg.paths import Paths


class DataTransformation:
    def __init__(self):
        self.config = Config()
        self.paths = Paths()
        self.artifacts_dir = self.paths.artifacts_dir

    def drop_unnecessary_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns=self.config.data_transformation.cols_to_drop, inplace=True)
        return df

    def replace_unknown_value_with_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        unknown_to_replace = ['?']
        for uk in unknown_to_replace:
            df = df.replace(uk, np.nan)
        return df

    def map_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_map = self.config.data_transformation.CategoricalFeatureMap
        for col, mapping in cat_map.__dict__.items():
            if col.startswith('_'):
                continue
            df[col] = df[col].map(mapping)
        return df

    def __get_dummies_for_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[col], inplace=True)
        return df

    def get_dummies_for_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df = self.__get_dummies_for_column(df, col)
        return df

    def apply_standard_scaler(self, df, is_train=False, is_pred=False):
        """
        Standard scaling the input df

        :param df: Input dataframe
        :param is_train: If True, fit_transform the train data, else transform the test/predict data
        :return: Scaled dataframe
        """
        try:
            scaler = StandardScaler()
            num_cols = self.config.data_transformation.numerical_columns
            num_df = df[num_cols]
            cols = num_df.columns

            artifacts_dir = self.paths.artifacts_dir
            std_scaler_pkl = self.paths.data_transformation.std_scaler_fname
            std_scaler_pkl_path = ""
            if is_pred:
                std_scaler_pkl_path = Path(artifacts_dir) / std_scaler_pkl
            else:
                std_scaler_pkl_path = Path("..") / ".." / artifacts_dir / std_scaler_pkl

            if is_train:

                scaled_df = scaler.fit_transform(num_df)
                dump(scaler, open(std_scaler_pkl_path, 'wb'))
                logging.info("StandardScaler object dumped after fit_transform")
            else:
                scaler = load(open(std_scaler_pkl_path, 'rb'))
                scaled_df = scaler.transform(num_df)
                logging.info("StandardScaler object loaded for transform")

            scaled_df = pd.DataFrame(scaled_df, columns=cols)

            return scaled_df

        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)

    def apply_knn_imputer(self, final_df, is_train=False, is_pred=False):
        """
        KNN imputation for missing data in categorical columns

        :param final_df: Input dataframe for imputation
        :param is_train: If True, fit_transform the train data, else transform the test/predict data
        :return: Imputed dataframe
        """
        try:
            imputer = KNNImputer(n_neighbors=self.config.data_transformation.knn_n_nbrs, missing_values=np.nan)

            cols = final_df.columns

            knn_imputer_pkl = self.paths.data_transformation.knn_imputer_fname
            knn_imputer_pkl_path = ""
            if is_pred:
                knn_imputer_pkl_path = Path(self.artifacts_dir) / knn_imputer_pkl
            else:
                knn_imputer_pkl_path = Path("..") / ".." / self.artifacts_dir / knn_imputer_pkl

            if is_train:
                imputed_df = imputer.fit_transform(final_df)
                dump(imputer, open(knn_imputer_pkl_path, 'wb'))
                logging.info("KNNImputer object dumped after fit_transform")
            else:
                imputer = load(open(knn_imputer_pkl_path, 'rb'))
                imputed_df = imputer.transform(final_df)
                logging.info("KNNImputer object loaded for transform")

            imputed_df = pd.DataFrame(imputed_df, columns=cols)

            return imputed_df

        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)

    def initiate_data_transformation(self, file_path: str, is_train: bool = False, is_pred: bool = False) -> str:
        """
        Transform the data in the given file_path and save the preprocessed file.
        If is_train is True, fit and transform the training data. If is_pred is True, transform the prediction data.
        Otherwise, transform the test data.

        Args:
            file_path (str): Path to the input file.
            is_train (bool, optional): Whether the file is training data. Defaults to False.
            is_pred (bool, optional): Whether the file is prediction data. Defaults to False.

        Returns:
            str: Path to the preprocessed file.
        """
        try:
            if is_train:
                logging.info("TRANSFORMATION FOR TRAIN DATA INITIATED")
            elif is_pred:
                logging.info("TRANSFORMATION FOR PRED DATA INITIATED")
            else:
                logging.info("TRANSFORMATION FOR TEST DATA INITIATED")

            logging.info("==============================================")

            complete_file_path = Path(file_path)
            df = pd.read_csv(complete_file_path)
            logging.info("Read of data completed")
            logging.info(f"Original shape - {df.shape}")

            df = self.drop_unnecessary_cols(df)
            logging.info("Deleted unnecessary columns")
            logging.info(f"After dropping unnecessary columns - {df.shape}")

            df = self.replace_unknown_value_with_nan(df)
            logging.info("Unknown values replaced with NaN")
            logging.info(f"After replacing unknown values with NaN - {df.shape}")

            df = self.map_categorical_features(df)
            logging.info("Custom mapping completed")
            logging.info(f"After custom mapping - {df.shape}")

            cat_df = self.get_dummies_for_categorical_columns(df)
            logging.info("Dummies Added")
            logging.info(f"After adding dummies - {cat_df.shape}")

            scaled_df = self.apply_standard_scaler(df, is_train, is_pred)
            logging.info("Standard Scaler Applied")
            logging.info(f"After applying standard scaler - {scaled_df.shape}")

            final_df = pd.concat([scaled_df, cat_df], axis=1)
            logging.info(f"Final df shape {final_df.shape}")

            imputed_df = self.apply_knn_imputer(final_df, is_train, is_pred)
            logging.info("KNN Imputer Applied")
            logging.info(f"After applying KNN imputer - {imputed_df.shape}")

            if is_train:
                preprocessed_train_fname = self.paths.data_transformation.preprocessed_train_fname
                preprocessed_train_path = Path("..") / ".." / self.artifacts_dir / preprocessed_train_fname
                imputed_df.to_csv(preprocessed_train_path, index=False)
                logging.info("TRANSFORMATION FOR TRAIN DATA COMPLETED")
                return str(preprocessed_train_path)
            elif is_pred:
                preprocessed_pred_fname = self.paths.data_transformation.preprocessed_pred_fname
                preprocessed_pred_path = Path(self.artifacts_dir) / preprocessed_pred_fname

                imputed_df.to_csv(preprocessed_pred_path, index=False)
                logging.info("TRANSFORMATION FOR PRED DATA COMPLETED")
                return str(preprocessed_pred_path)
            else:
                preprocessed_test_fname = self.paths.data_transformation.preprocessed_test_fname
                preprocessed_test_path = Path("..") / ".." / self.artifacts_dir / preprocessed_test_fname

                imputed_df.to_csv(preprocessed_test_path, index=False)
                logging.info("TRANSFORMATION FOR TEST DATA COMPLETED")
                return str(preprocessed_test_path)

        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
