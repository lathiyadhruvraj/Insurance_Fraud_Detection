import os
import sys

from src.logger import logging

import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
import pandas as pd

def read_raw_data(file_path):
    """
    Reads the raw data file and returns it as a pandas dataframe.
    """
    with open(file_path) as f:
        df = pd.read_csv(f)

    return df

def write_raw_data(df, file_path):
    """
    Writes the raw data dataframe to a CSV file.
    """
    with open(file_path, 'w') as f:
        df.to_csv(f, index=False, header=True)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param, cv):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=cv)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_roc_auc_score = roc_auc_score(y_train, y_train_pred)
            test_model_roc_auc_score = roc_auc_score(y_test, y_test_pred)

            train_model_acc_score = accuracy_score(y_train, y_train_pred)
            test_model_acc_score = accuracy_score(y_test, y_test_pred)

            train_model_f1_score = f1_score(y_train, y_train_pred)
            test_model_f1_score = f1_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_f1_score

            logging.info(f"{model} : Train : roc_auc - {train_model_roc_auc_score}  acc -  {train_model_acc_score} f1 - {train_model_f1_score}")

            logging.info(f"{model}  : Test : roc_auc - {test_model_roc_auc_score} acc - {test_model_acc_score} f1 - {test_model_f1_score}")

        return report

    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.exception(e)
        raise CustomException(e, sys)