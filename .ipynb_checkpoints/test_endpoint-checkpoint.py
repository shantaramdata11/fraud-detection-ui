import boto3

# Endpoint configuration
ENDPOINT_NAME = "fraud-detection-endpoint-v1"
REGION = "ap-south-1"

# Create runtime client
runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION
)

# Credit card dataset has 29 features: V1–V28 + Amount
# Generate a sample transaction
test_data = ",".join(["1.0"] * 28 + ["100.0"])

try:
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=test_data
    )

    result = response["Body"].read().decode()

    print("🤖 Model Prediction:", result)

except Exception as e:
    print("❌ Error:", str(e))