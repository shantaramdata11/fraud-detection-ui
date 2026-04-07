import os
import joblib
import numpy as np

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        return request_body.strip()
    raise ValueError("Unsupported content type")

def predict_fn(input_data, model):
    data = np.fromstring(input_data, sep=',')
    return model.predict(data.reshape(1, -1))

def output_fn(prediction, content_type):
    return str(prediction[0])
