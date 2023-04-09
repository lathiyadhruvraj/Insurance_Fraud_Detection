from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_dir: str = 'Data'
    raw_data_file: str = 'insuranceFraud.csv'
    raw_data_cpy: str = 'data_copy.csv'
    train_data_file: str = 'train.csv'
    test_data_file: str = 'test.csv'

@dataclass(frozen=True)
class DataTransformationConfig:
    std_scaler_fname: str = 'std_scaler.pkl'
    knn_imputer_fname: str = 'knn_imputer.pkl'
    preprocessed_train_fname: str = 'train_preprocessed.csv'
    preprocessed_test_fname: str = 'test_preprocessed.csv'
    preprocessed_pred_fname: str = 'prediction_preprocessed.csv'

@dataclass(frozen=True)
class ModelTrainerConfig:
    trained_model_fname: str = 'model.pkl'

@dataclass(frozen=True)
class Paths:
    data_ingestion: DataIngestionConfig = DataIngestionConfig()
    data_transformation: DataTransformationConfig = DataTransformationConfig()
    model_trainer: ModelTrainerConfig = ModelTrainerConfig()
    artifacts_dir: str = 'artifacts'

