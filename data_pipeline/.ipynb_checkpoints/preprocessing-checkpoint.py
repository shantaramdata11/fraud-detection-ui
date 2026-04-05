import pandas as pd
import boto3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import os

# --- Industry Standard: S3 Path Configuration ---
S3_INPUT_PATH = 's3://fraud-ml-dev/raw/creditcard.csv'
S3_OUTPUT_TRAIN = 's3://fraud-ml-dev/curated/train.csv'
S3_OUTPUT_TEST = 's3://fraud-ml-dev/curated/test.csv'

def preprocess():
    print(f"🚀 Streaming data directly from Data Lake: {S3_INPUT_PATH}...")
    
    # 1. Read directly from S3 (Requires s3fs installed)
    try:
        df = pd.read_csv(S3_INPUT_PATH)
    except Exception as e:
        print(f"❌ Connection Error: Ensure s3fs is installed and IAM role has S3 access. \nDetail: {e}")
        return

    # 2. Senior Practice: Feature Scaling
    # Using RobustScaler to handle fraud outliers without skewing the distribution
    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    
    # Drop original features as they are now redundant
    df.drop(['Time','Amount'], axis=1, inplace=True)
    
    # 3. Stratified Split (80/20)
    # Stratification ensures the 0.17% fraud ratio is preserved in both sets
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Class']
    )
    
    # 4. Write directly back to S3 Curated Layer
    print(f"📤 Writing processed artifacts to S3 Curated Layer...")
    train.to_csv(S3_OUTPUT_TRAIN, index=False)
    test.to_csv(S3_OUTPUT_TEST, index=False)
    
    print(f"✅ Pipeline Success: {S3_INPUT_PATH} -> {os.path.dirname(S3_OUTPUT_TRAIN)}")

if __name__ == '__main__':
    preprocess()