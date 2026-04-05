import boto3
import sagemaker
from sagemaker.xgboost.model import XGBoostModel

# 1. Setup
role = sagemaker.get_execution_role()
bucket = "fraud-ml-dev"
endpoint_name = "fraud-detection-endpoint-v1"
sm_client = boto3.client('sagemaker')

print("🧹 Cleaning up the failed containers...")
for func in [sm_client.delete_endpoint, sm_client.delete_endpoint_config]:
    try:
        if func == sm_client.delete_endpoint:
            func(EndpointName=endpoint_name)
        else:
            func(EndpointConfigName=endpoint_name)
    except:
        pass

# 2. XGBOOST DEPLOYMENT (The Correct Engine)
print("🚀 Launching XGBOOST Native Deployment...")
model = XGBoostModel(
    model_data=f"s3://{bucket}/model/model.tar.gz",
    role=role,
    entry_point="inference.py", # Use the flat inference.py we made
    framework_version="1.7-1",   # High-performance XGBoost version
)

# Using the cheaper instance as recommended by your console
model.deploy(
    initial_instance_count=1,
    instance_type="ml.c6i.large", 
    endpoint_name=endpoint_name
)

print(f"\n🎉 SUCCESS! XGBoost Endpoint {endpoint_name} is LIVE.")