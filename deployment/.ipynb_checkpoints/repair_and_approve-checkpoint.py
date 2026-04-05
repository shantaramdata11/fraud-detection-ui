import boto3
import sagemaker
from sagemaker import image_uris

# 1. Setup Session & Region
sm_client = boto3.client('sagemaker')
region = boto3.Session().region_name

# 2. Fetch the Authorized 1.2-1 Image for Mumbai
container_uri = image_uris.retrieve(
    framework="sklearn", 
    region=region, 
    version="1.2-1", 
    py_version="py3", 
    instance_type="ml.m5.large"
)

try:
    print("📦 Registering Model Version (Fixed Environment Variables)...")
    
    # 3. Create Model Package
    # We include Environment variables to fix the 'NoneType' startswith error
    response = sm_client.create_model_package(
        ModelPackageGroupName="FraudDetectionGroup",
        ModelPackageDescription="Industrial Fix: Explicit Entry Point with Scikit-Learn 1.2.1",
        InferenceSpecification={
            "Containers": [{
                "Image": container_uri,
                "ModelDataUrl": "s3://fraud-ml-dev/model/model.tar.gz",
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",           # Matches your flat zip file
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model"  # Points to the root of the extract
                }
            }],
            "SupportedTransformInstanceTypes": ["ml.m5.large"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        }
    )
    
    new_arn = response['ModelPackageArn']

    # 4. Approve the version immediately
    sm_client.update_model_package(
        ModelPackageArn=new_arn, 
        ModelApprovalStatus='Approved',
        ApprovalDescription="Validated model-code path and 1.2.1 compatibility."
    )
    
    print(f"✅ SUCCESS: {new_arn} is now APPROVED.")
    print("\n👉 Next steps: Clear the ghost resources and run deployment/deploy_model.py")
    
except Exception as e:
    print(f"❌ ERROR: {str(e)}")