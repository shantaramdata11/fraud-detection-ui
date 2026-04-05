import boto3
import sagemaker
from sagemaker import image_uris
import os
import time

# 1. Setup
sm_client = boto3.client('sagemaker')
s3 = boto3.client('s3')
region = boto3.Session().region_name
role = sagemaker.get_execution_role()
bucket = "fraud-ml-dev"
endpoint_name = "fraud-detection-endpoint-v1"

print("🛠️  Step 1: Creating Flat Inference Script...")
with open("inference.py", "w") as f:
    f.write("""import os
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
""")

print("📦 Step 2: Packaging model.tar.gz...")
os.system("tar -czvf model.tar.gz model.joblib inference.py")

print("📤 Step 3: Uploading to S3...")
s3.upload_file("model.tar.gz", bucket, "model/model.tar.gz")

print("📦 Step 4: Registering Version 7 with MANDATORY Env Variables...")
container_uri = image_uris.retrieve(
    framework="sklearn", region=region, version="1.2-1", py_version="py3", instance_type="ml.m5.large"
)

response = sm_client.create_model_package(
    ModelPackageGroupName="FraudDetectionGroup",
    ModelPackageDescription="Final Fix: Explicit Entry Point",
    InferenceSpecification={
        "Containers": [{
            "Image": container_uri,
            "ModelDataUrl": f"s3://{bucket}/model/model.tar.gz",
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",           # THE MISSING PIECE
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model"  # THE MISSING PIECE
            }
        }],
        "SupportedTransformInstanceTypes": ["ml.m5.large"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"],
    }
)
new_arn = response['ModelPackageArn']
sm_client.update_model_package(ModelPackageArn=new_arn, ModelApprovalStatus='Approved')
print(f"✅ Version 7 Approved: {new_arn}")

print("🗑️  Step 5: Cleaning up failed resources...")
try:
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    time.sleep(5)
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    time.sleep(5)
except:
    pass

print("🚀 Step 6: Deploying Final Infrastructure...")
model = sagemaker.ModelPackage(
    role=role,
    model_package_arn=new_arn,
    sagemaker_session=sagemaker.Session()
)

model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=endpoint_name
)

print(f"\n🎉 FINALLY! Endpoint {endpoint_name} is LIVE.")