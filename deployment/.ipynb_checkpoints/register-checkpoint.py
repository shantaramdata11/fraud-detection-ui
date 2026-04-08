import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# ---------------------------
# Setup
# ---------------------------

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

S3_BUCKET = "fraud-ml-dev"
S3_MODEL_PREFIX = "model/"
MODEL_GROUP_NAME = "FraudDetectionGroup"

s3_client = boto3.client("s3")


# ---------------------------
# Get Latest Model Artifact
# ---------------------------

def get_latest_model():

    print("🔍 Searching for latest model artifact in S3...")

    response = s3_client.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=S3_MODEL_PREFIX
    )

    if "Contents" not in response:
        raise Exception("❌ No model artifacts found")

    # Filter only tar.gz models
    artifacts = [
        obj for obj in response["Contents"]
        if obj["Key"].endswith(".tar.gz")
    ]

    if not artifacts:
        raise Exception("❌ No .tar.gz model artifacts found")

    latest_object = sorted(
        artifacts,
        key=lambda x: x["LastModified"],
        reverse=True
    )[0]

    latest_key = latest_object["Key"]

    latest_model_path = f"s3://{S3_BUCKET}/{latest_key}"

    print(f"✅ Latest model found: {latest_model_path}")

    return latest_model_path


# ---------------------------
# Register Model
# ---------------------------

def register_fraud_model():

    model_s3_path = get_latest_model()

    print(f"🚀 Registering model: {model_s3_path}")

    model = SKLearnModel(
        model_data=model_s3_path,
        role=role,
        entry_point="inference.py",          # REQUIRED
        source_dir=".",                      # REQUIRED
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sagemaker_session
    )

    model_package = model.register(
        content_types=["text/csv"],
        response_types=["application/json"],
        model_package_group_name=MODEL_GROUP_NAME,
        approval_status="PendingManualApproval",
        description="XGBoost Fraud Detection Model"
    )

    print("\n✅ Model registered successfully")
    print(f"📦 Model Package ARN: {model_package.model_package_arn}")


if __name__ == "__main__":
    register_fraud_model()