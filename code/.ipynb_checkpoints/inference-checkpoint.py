import os
import joblib
import numpy as np
import json

def model_fn(model_dir):
    """SageMaker calls this to load the model"""
    # In a flat structure, the file is directly in model_dir
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        print(f"❌ ERROR: File not found at {model_path}")
        # List files to help debug in logs
        print(f"📂 Available files: {os.listdir(model_dir)}")
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        return request_body.strip()
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    data = np.fromstring(input_data, sep=',')
    return model.predict(data.reshape(1, -1))

def output_fn(prediction, content_type):
    return str(prediction[0])