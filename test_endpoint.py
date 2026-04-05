import boto3
import json

runtime = boto3.client('sagemaker-runtime')

# Exactly 29 features (No spaces, pure CSV)
# This represents [V1, V2, ... V28, Amount]
test_data = "1.0," * 28 + "100.0" 

try:
    response = runtime.invoke_endpoint(
        EndpointName='fraud-detection-endpoint-v1',
        ContentType='text/csv',
        Body=test_data
    )
    result = response['Body'].read().decode()
    print(f"🤖 Model Prediction: {result}")
except Exception as e:
    print(f"❌ Error: {str(e)}")