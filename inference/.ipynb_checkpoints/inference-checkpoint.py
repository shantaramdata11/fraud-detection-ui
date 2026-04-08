import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

S3_BUCKET = "fraud-ml-dev"
S3_MODEL_PREFIX = "model/"
MODEL_GROUP_NAME = "FraudDetectionGroup"

s3 = boto3.client("s3")


def get_latest_model():
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_MODEL_PREFIX)

    artifacts = [obj for obj in response["Contents"] if obj["Key"].endswith(".tar.gz")]

    latest = sorted(artifacts, key=lambda x: x["LastModified"], reverse=True)[0]

    return f"s3://{S3_BUCKET}/{latest['Key']}"


def register_fraud_model():
    model_s3_path = get_latest_model()

    print(f"Registering model from {model_s3_path}")

    model = SKLearnModel(
        model_data=model_s3_path,
        role=role,
        entry_point="inference.py",      # critical fix
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sagemaker_session
    )

    model_package = model.register(
        content_types=["text/csv"],
        response_types=["application/json"],
        model_package_group_name=MODEL_GROUP_NAME,
        approval_status="PendingManualApproval",
        description="Fraud Detection Model"
    )

    print("Model registered:", model_package.model_package_arn)


if __name__ == "__main__":
    register_fraud_model()