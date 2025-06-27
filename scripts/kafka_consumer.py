from kafka import KafkaConsumer
import joblib
import json
import numpy as np
import pandas as pd
import os

model = joblib.load('../models/xgb_model.pkl')

consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='fraud_detection_group'
)

output_path = 'output/fraud_log.csv'
os.makedirs('output', exist_ok = True)

if not os.path.exists(output_path):
    df = pd.DataFrame(columns=['prediction', 'probability', 'Transaction']).to_csv(output_path, index=False)

print("📡 Listening for transactions...")

feature_columns = [
    'Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 
    'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 
    'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

for message in consumer:
    transaction = message.value
    features = np.array([[transaction[col] for col in feature_columns]])
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    print(f"Transaction: {transaction}")
    print(f"🔥 Fraud Prediction: {prediction} | Probability: {round(probability, 4)}")
    print("=" * 50) 

    log_entry={
        'prediction': int(prediction),
        'probability': round(probability, 4),
        'Transaction': json.dumps(transaction)
    }

    log_df = pd.DataFrame([log_entry])
    log_df.to_csv(output_path, mode='a', header=False, index=False)
    print(f"Logged fraud={prediction} (prob={probability:.4f})")


