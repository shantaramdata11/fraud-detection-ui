import pandas as pd
import xgboost as xgb
import joblib
import tarfile
import boto3
from sklearn.metrics import classification_report
from datetime import datetime

# -------------------------------
# S3 CONFIGURATION
# -------------------------------

S3_INPUT_TRAIN = "s3://fraud-ml-dev/curated/train.csv"
S3_INPUT_TEST = "s3://fraud-ml-dev/curated/test.csv"

S3_BUCKET = "fraud-ml-dev"
S3_MODEL_PREFIX = "model/"


def train():

    print("🚀 Loading curated datasets from S3...")

    # Load training data
    train_df = pd.read_csv(S3_INPUT_TRAIN)
    test_df = pd.read_csv(S3_INPUT_TEST)

    X_train = train_df.drop("Class", axis=1)
    y_train = train_df["Class"]

    X_test = test_df.drop("Class", axis=1)
    y_test = test_df["Class"]

    # -------------------------------
    # HANDLE CLASS IMBALANCE
    # -------------------------------

    num_neg = len(y_train[y_train == 0])
    num_pos = len(y_train[y_train == 1])

    scale_pos_weight = num_neg / num_pos

    print(f"⚖️ Calculated scale_pos_weight = {scale_pos_weight:.2f}")

    # -------------------------------
    # TRAIN MODEL
    # -------------------------------

    print("🏗️ Training XGBoost classifier...")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr"
    )

    model.fit(X_train, y_train)

    # -------------------------------
    # EVALUATION
    # -------------------------------

    print("\n📊 Model Evaluation")

    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    # -------------------------------
    # SAVE MODEL LOCALLY
    # -------------------------------

    print("\n💾 Saving trained model...")

    joblib.dump(model, "model.joblib")

    # -------------------------------
    # CREATE VERSIONED ARTIFACT
    # -------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = f"model_{timestamp}.tar.gz"

    print(f"📦 Packaging model artifact: {model_filename}")

    with tarfile.open(model_filename, "w:gz") as tar:
        tar.add("model.joblib")
        tar.add("inference/inference.py", arcname="inference.py")  # IMPORTANT

    # -------------------------------
    # UPLOAD TO S3
    # -------------------------------

    print("☁️ Uploading artifact to S3...")

    s3 = boto3.client("s3")

    s3_key = f"{S3_MODEL_PREFIX}{model_filename}"

    s3.upload_file(model_filename, S3_BUCKET, s3_key)

    print(f"✅ Model uploaded to s3://{S3_BUCKET}/{s3_key}")


if __name__ == "__main__":
    train()