import boto3
import sagemaker
from sagemaker import get_execution_role
import time

# ---------------------------
# Setup
# ---------------------------

sagemaker_session = sagemaker.Session()
role = get_execution_role()

sm_client = boto3.client("sagemaker")

MODEL_GROUP_NAME = "FraudDetectionGroup"
ENDPOINT_NAME = "fraud-detection-endpoint-v1"


def get_latest_approved_model():

    print("🔍 Searching for latest APPROVED model...")

    response = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_GROUP_NAME,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending"
    )

    if not response["ModelPackageSummaryList"]:
        raise Exception("❌ No approved model found")

    model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]

    print(f"✅ Found approved model: {model_package_arn}")

    return model_package_arn


def delete_old_endpoint():

    for resource in ["Endpoint", "EndpointConfig"]:

        try:

            if resource == "Endpoint":
                print(f"🗑️ Deleting existing endpoint {ENDPOINT_NAME}")
                sm_client.delete_endpoint(EndpointName=ENDPOINT_NAME)

            else:
                print(f"🗑️ Deleting existing endpoint config {ENDPOINT_NAME}")
                sm_client.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)

            time.sleep(5)

        except sm_client.exceptions.ClientError:
            print(f"ℹ️ No existing {resource}")


def deploy():

    model_package_arn = get_latest_approved_model()

    delete_old_endpoint()

    model = sagemaker.ModelPackage(
        role=role,
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session
    )

    print("🚀 Deploying model endpoint (ml.m5.large)...")

    model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME
    )

    print(f"\n🎉 SUCCESS — Endpoint {ENDPOINT_NAME} is LIVE")


if __name__ == "__main__":
    deploy()