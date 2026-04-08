import boto3

sm_client = boto3.client("sagemaker")

MODEL_GROUP_NAME = "FraudDetectionGroup"


def approve_model():

    print("🔍 Finding latest model version for approval...")

    response = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_GROUP_NAME,
        SortBy="CreationTime",
        SortOrder="Descending"
    )

    if not response["ModelPackageSummaryList"]:
        print("❌ No models found in registry")
        return

    latest_model = response["ModelPackageSummaryList"][0]

    model_package_arn = latest_model["ModelPackageArn"]

    print(f"🕵️ Approving model: {model_package_arn}")

    sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus="Approved",
        ApprovalDescription="Auto approval for deployment pipeline"
    )

    print("✅ Model approved successfully")


if __name__ == "__main__":
    approve_model()