import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file:str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformer_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaling", StandardScaler(with_mean=True))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_cols),
                ]
            )

            return preprocessor
    
        except Exception as e:
            raise CustomException(e, sys)    
        
    def initiate_data_transformation(self, train_data, test_data):
        try:
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            logging.info("Read train and test data Completed")

            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()

            target_feature = "label"

            input_feature_train_df = train_df.drop(columns=[target_feature], axis=1)
            target_feature_train_df = train_df[target_feature]


            input_feature_test_df = test_df.drop(columns=[target_feature], axis=1)
            target_feature_test_df = test_df[target_feature]

            logging.info(
                "Applying preprocessing object on training dataFrame and testing Dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            label_encoder = LabelEncoder()

            target_feature_train_encoded = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_encoded = label_encoder.transform(target_feature_test_df)

            train_arr = np.concatenate(
            [input_feature_train_arr, target_feature_train_encoded.reshape(-1, 1)],
            axis=1
            )

            test_arr = np.concatenate(
            [input_feature_test_arr, target_feature_test_encoded.reshape(-1, 1)],
            axis=1
            )

            logging.info("saved Preprocessing Object")

            save_object(
                file_path=self.data_transformer_config.preprocessor_obj_file,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_file
            )

        except Exception as e:
            raise CustomException(e, sys)