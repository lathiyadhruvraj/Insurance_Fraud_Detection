from dataclasses import dataclass, field
from typing import List
from typing import Dict

import os

@dataclass
class StreamlitConfig:
    bg_url : str =  "https://cdn.pixabay.com/photo/2020/04/20/04/02/brick-5066282_960_720.jpg"
    title : str = "Insurance Fraud Detection"
    files_dir : str = "predict_files"
    artifacts_dir: str = 'artifacts'

@dataclass(frozen=True)
class DataIngestionConfig:
    ttsplit_test_size: float = 0.25
    ttsplit_random_state: int = 42

def default_cols_to_drop() -> List[str]:
    return ['policy_number', 'policy_bind_date', 'policy_state', 'insured_zip', 'incident_location',
            'incident_date',  'incident_state', 'incident_city', 'insured_hobbies',
            'auto_make', 'auto_model', 'auto_year', 'age', 'total_claim_amount']

def default_custom_mapping_columns() -> List[str]:
    return ['policy_csl', 'insured_education_level', 'incident_severity', 'insured_sex',
                        'property_damage', 'police_report_available', 'fraud_reported']

def default_dummies_columns() -> List[str]:
    return ['insured_occupation', 'insured_relationship', 'incident_type', 'collision_type',
                       'authorities_contacted']

def default_numerical_columns() -> List[str]:
    return ['months_as_customer', 'policy_deductable', 'umbrella_limit', 'capital-gains',
            'capital-loss', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
            'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim', 'vehicle_claim']

@dataclass(frozen=True)
class DataTransformationConfig:
    knn_n_nbrs: int = 3
    cols_to_drop: List[str] = field(default_factory=default_cols_to_drop)
    custom_mapping_columns: List[str] = field(default_factory=default_custom_mapping_columns)
    dummies_columns: List[str] = field(default_factory=default_dummies_columns)
    numerical_columns: List[str] = field(default_factory=default_numerical_columns)

    @dataclass
    class CategoricalFeatureMap:
        policy_csl: Dict[str, float] = field(default_factory=lambda: {'100/300': 1, '250/500': 2.5, '500/1000': 5})
        insured_education_level: Dict[str, int] = field(default_factory=lambda: {'JD': 1, 'High School': 2, 'College': 3, 'Masters': 4, 'Associate': 5, 'MD': 6, 'PhD': 7})
        incident_severity: Dict[str, int] = field(default_factory=lambda: {'Trivial Damage': 1, 'Minor Damage': 2, 'Major Damage': 3, 'Total Loss': 4})
        insured_sex: Dict[str, int] = field(default_factory=lambda: {'FEMALE': 0, 'MALE': 1})
        property_damage: Dict[str, int] = field(default_factory=lambda: {'NO': 0, 'YES': 1})
        police_report_available: Dict[str, int] = field(default_factory=lambda: {'NO': 0, 'YES': 1})
        fraud_reported: Dict[str, int] = field(default_factory=lambda: {'N': 0, 'Y': 1})

@dataclass(frozen=True)
class ModelTrainerConfig:
    best_model_score_threshold: float = 0.56
    cv: int = 5


@dataclass(frozen=True)
class Config:
    data_ingestion: DataIngestionConfig = DataIngestionConfig()
    data_transformation: DataTransformationConfig = DataTransformationConfig()
    model_trainer: ModelTrainerConfig = ModelTrainerConfig()
    artifacts_dir: str = 'artifacts'



