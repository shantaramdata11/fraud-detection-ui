import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# 1. Setup Session and Variables
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = sagemaker_session.boto_region_name

# These must match your S3 paths and Group Name
MODEL_S3_PATH = "s3://fraud-ml-dev/model/model.tar.gz"
MODEL_GROUP_NAME = "FraudDetectionGroup"

def register_fraud_model():
    print(f"🚀 Starting registration for: {MODEL_S3_PATH}")

    # 2. Create the Model Object
    # We use the pre-built SageMaker Scikit-Learn container
    model = SKLearnModel(
        model_data=MODEL_S3_PATH,
        role=role,
        framework_version="1.2-1",  # Ensure this matches your training environment
        py_version="py3",
        sagemaker_session=sagemaker_session
    )

    # 3. Register the Model Version
    try:
        model_package = model.register(
            content_types=["text/csv"],
            response_types=["application/json"],
            model_package_group_name=MODEL_GROUP_NAME,
            image_uri=None,  # SageMaker infers the image from SKLearnModel class
            approval_status="PendingManualApproval",  # Industry standard for governance
            description="XGBoost Fraud Detection Model - Version 1"
        )
        
        print(f"✅ Success! Model registered in group: {MODEL_GROUP_NAME}")
        print(f"Model Package ARN: {model_package.model_package_arn}")
        
    except Exception as e:
        print(f"❌ Error during registration: {str(e)}")
        print("Tip: Check if the Model Group 'FraudDetectionGroup' exists in the UI first.")

if __name__ == "__main__":
    register_fraud_model()