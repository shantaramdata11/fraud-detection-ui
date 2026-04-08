import os
import boto3
import json

# Initialize the SageMaker Runtime client
runtime = boto3.client('sagemaker-runtime')
# This name must match exactly what is in your DIRECT_DEPLOY script
ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME', 'fraud-detection-endpoint-v1')

def lambda_handler(event, context):
    print(f"📥 Incoming Event: {json.dumps(event)}")
    
    try:
        # 1. Extract raw data (Expects a CSV string of features)
        payload = event['data']
        
        # 2. Call the SageMaker Endpoint
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=payload
        )
        
        # 3. Parse the result from the model
        result = response['Body'].read().decode()
        prediction = int(float(result))
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': prediction,
                'status': '🚨 FRAUD DETECTED' if prediction == 1 else '✅ LEGITIMATE TRANSACTION',
                'model_endpoint': ENDPOINT_NAME
            })
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Prediction failed', 'details': str(e)})
        }