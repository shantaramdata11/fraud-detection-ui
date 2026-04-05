import boto3

# Initialize the SageMaker client
sm_client = boto3.client('sagemaker')

# This is the exact ARN from your successful registration terminal output
MODEL_PACKAGE_ARN = "arn:aws:sagemaker:ap-south-1:306005334235:model-package/FraudDetectionGroup/2"

def approve_model():
    print(f"🕵️  Manually approving model version: {MODEL_PACKAGE_ARN}")
    
    try:
        sm_client.update_model_package(
            ModelPackageArn=MODEL_PACKAGE_ARN,
            ModelApprovalStatus='Approved',
            ApprovalDescription="Senior Engineer manual approval to unblock production deployment."
        )
        print("✅ SUCCESS: Model status updated to APPROVED.")
    except Exception as e:
        print(f"❌ ERROR: Could not update status. Details: {str(e)}")

if __name__ == "__main__":
    approve_model()