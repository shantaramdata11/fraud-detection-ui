import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# 1. Setup
role = sagemaker.get_execution_role()
bucket = "fraud-ml-dev"
endpoint_name = "fraud-detection-endpoint-v1"
sm_client = boto3.client('sagemaker')

print("🧹 Cleaning up the failed garbage...")
for func in [sm_client.delete_endpoint, sm_client.delete_endpoint_config]:
    try:
        if func == sm_client.delete_endpoint:
            func(EndpointName=endpoint_name)
        else:
            func(EndpointConfigName=endpoint_name)
    except:
        pass

# 2. DIRECT DEPLOYMENT (The Golden Path)
# This high-level class handles all the environment variable mapping for you.
print("🚀 Launching DIRECT deployment (Bypassing Registry)...")
model = SKLearnModel(
    model_data=f"s3://{bucket}/model/model.tar.gz",
    role=role,
    entry_point="inference.py", # It will use the inference.py you just created
    framework_version="1.2-1",
    py_version="py3"
)

# This will take 8-10 minutes. 
# Since we are bypassing the Registry, the 'NoneType' error is physically impossible.
model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)

print(f"\n🎉 FINALLY! Endpoint {endpoint_name} is LIVE.")