import sys
import pandas as pd
import os

from src.exception import CustomException
from src.utils import load_object

class predictPipeline:
    def __init__(self):
        pass

    def predict(self, Dataframe):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(Dataframe)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        N: int,
        P: int,
        K: int,
        temperature:float,
        humidity: float,
        ph: float,
        rainfall: float):

        self.N = N
        self.P = P
        self.K = K
        self.temperature = temperature
        self.humidity = humidity
        self.ph = ph
        self.rainfall = rainfall

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
            "N": [self.N],
            "P": [self.P],
            "K": [self.K],
            "temperature": [self.temperature],
            "humidity": [self.humidity],
            "ph": [self.ph],
            "rainfall": [self.rainfall]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)