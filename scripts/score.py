import json
import pandas as pd
import os
import joblib, pickle
from azureml.core import Model


def init():
    global model
    model_path = Model.get_model_path('best-model-machine-cpu-hd')

    model = joblib.load(model_path)

def run(data):
    try:
        input_data = json.loads(data)
        data = pd.DataFrame(input_data['data'])
        result = model.predict(data)
        # return serialisable result.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error