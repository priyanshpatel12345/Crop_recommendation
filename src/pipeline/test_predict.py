from src.pipeline.predict_pipeline import predictPipeline, CustomData
from src.utils import load_object

import os

label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")
label_encoder = load_object(label_encoder_path)


custom_data = CustomData(
    N=10,
    P=22,
    K=13,
    temperature=25.5,
    humidity=65.2,
    ph=4.8,
    rainfall=120.0
)

df = custom_data.get_data_as_dataframe()
print("DataFrame: ", df)

pipeline = predictPipeline()
prediction = pipeline.predict(df)
# print("Output:", prediction)

original_label = label_encoder.inverse_transform(prediction.astype(int))
print("Output:", original_label)