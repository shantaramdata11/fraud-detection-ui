import boto3
import sagemaker
from sagemaker import get_execution_role
import time

# 1. Setup Session & Clients
sagemaker_session = sagemaker.Session()
role = get_execution_role()
sm_client = boto3.client('sagemaker')

# 2. Configuration
MODEL_GROUP_NAME = "FraudDetectionGroup"
ENDPOINT_NAME = "fraud-detection-endpoint-v1"

def deploy():
    print(f"🔍 Finding latest Approved version in {MODEL_GROUP_NAME}...")
    
    # 3. Fetch the Latest Approved Version ( picks up Version 4 )
    approved_packages = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_GROUP_NAME,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending'
    )
    
    if not approved_packages['ModelPackageSummaryList']:
        print("❌ No Approved version found!")
        return

    model_package_arn = approved_packages['ModelPackageSummaryList'][0]['ModelPackageArn']
    print(f"✅ Found Approved Model: {model_package_arn}")

    # 4. SCORCHED EARTH: Force-delete existing endpoint and config
    for resource_type in ["Endpoint", "EndpointConfig"]:
        try:
            print(f"🗑️  Cleaning up old {resource_type}: {ENDPOINT_NAME}")
            if resource_type == "Endpoint":
                sm_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
            else:
                sm_client.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)
            
            # Wait for AWS to actually release the name
            print(f"⏳ Waiting for {resource_type} deletion...")
            time.sleep(5) 
        except sm_client.exceptions.ClientError:
            print(f"ℹ️  No existing {resource_type} to delete.")

    # 5. Create Model Object
    model = sagemaker.ModelPackage(
        role=role,
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session
    )

    # 6. Launch Infrastructure (ml.m5.large for ap-south-1)
    print(f"🚀 Deploying to ml.m5.large... (Wait ~8 mins)")
    model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large', 
        endpoint_name=ENDPOINT_NAME
    )
    print(f"\n🎉 SUCCESS! Endpoint {ENDPOINT_NAME} is LIVE.")

if __name__ == "__main__":
    deploy()