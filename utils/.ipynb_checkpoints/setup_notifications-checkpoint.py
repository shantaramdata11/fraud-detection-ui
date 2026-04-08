import boto3
import json

# Configuration
REGION = "ap-south-1"
EMAIL = "shantaram.data11@gmail.com"
TOPIC_NAME = "FraudModelRegistryAlerts"
MODEL_GROUP = "FraudDetectionGroup"

sns = boto3.client("sns", region_name=REGION)
events = boto3.client("events", region_name=REGION)

def setup():
    # 1. Create SNS Topic
    print(f"📡 Creating SNS Topic: {TOPIC_NAME}...")
    topic = sns.create_topic(Name=TOPIC_NAME)
    topic_arn = topic["TopicArn"]

    # 2. Subscribe your email
    print(f"📧 Subscribing {EMAIL} to topic...")
    sns.subscribe(
        TopicArn=topic_arn,
        Protocol="email",
        Endpoint=EMAIL
    )

    # 3. Create EventBridge Rule (The Listener)
    rule_name = "NotifyOnModelRegistration"
    event_pattern = {
        "source": ["aws.sagemaker"],
        "detail-type": ["SageMaker Model Package State Change"],
        "detail": {
            "ModelPackageGroupName": [MODEL_GROUP]
        }
    }

    print(f"👂 Creating EventBridge Rule: {rule_name}...")
    events.put_rule(
        Name=rule_name,
        EventPattern=json.dumps(event_pattern),
        State="ENABLED",
        Description=f"Notify {EMAIL} when a model is registered in {MODEL_GROUP}"
    )

    # 4. Add SNS as the Target for the Rule
    events.put_targets(
        Rule=rule_name,
        Targets=[{"Id": "SendEmail", "Arn": topic_arn}]
    )

    # 5. Permission: Allow EventBridge to publish to SNS
    # Get your account ID from the IAM role ARN or context
    account_id = "306005334235" # Your ID from the terminal screenshot

    sns.add_permission(
        TopicArn=topic_arn,
        Label="AllowEventBridgeToPublish",
        AWSAccountId=[account_id], 
        ActionName=["Publish"] # REMOVED "SNS:" prefix
    )

    print("\n✅ SETUP COMPLETE!")
    print(f"⚠️ ACTION REQUIRED: Check {EMAIL} and click 'Confirm Subscription' in the AWS email.")

if __name__ == "__main__":
    setup()