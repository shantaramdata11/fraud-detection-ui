import pandas as pd
import xgboost as xgb
import joblib
import boto3
from sklearn.metrics import classification_report, roc_auc_score
import os

# --- S3 Configuration ---
S3_INPUT_TRAIN = 's3://fraud-ml-dev/curated/train.csv'
S3_INPUT_TEST = 's3://fraud-ml-dev/curated/test.csv'
S3_BUCKET = 'fraud-ml-dev'
S3_MODEL_KEY = 'model/fraud_model.joblib'

def train():
    print(f"🚀 Loading curated data from S3...")
    
    # 1. Load Data directly from S3
    train_df = pd.read_csv(S3_INPUT_TRAIN)
    test_df = pd.read_csv(S3_INPUT_TEST)
    
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']

    # 2. Handle Imbalance: Calculate Scale_Pos_Weight
    # This tells XGBoost to care more about the rare fraud cases
    num_neg = len(y_train[y_train == 0])
    num_pos = len(y_train[y_train == 1])
    spw = num_neg / num_pos
    print(f"⚖️ Scale_Pos_Weight: {spw:.2f}")

    # 3. Train Model
    print("🏗️ Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=spw,
        eval_metric='aucpr' 
    )
    model.fit(X_train, y_train)

    # 4. Evaluate
    preds = model.predict(X_test)
    print("\n📊 Model Evaluation:")
    print(classification_report(y_test, preds))

    # 5. Upload Artifact to S3 Model Layer
    print("📤 Uploading model to S3...")
    joblib.dump(model, 'model.joblib')
    
    s3 = boto3.client('s3')
    s3.upload_file('model.joblib', S3_BUCKET, S3_MODEL_KEY)
    
    print(f"✅ Success! Model saved to s3://{S3_BUCKET}/{S3_MODEL_KEY}")

if __name__ == '__main__':
    train()