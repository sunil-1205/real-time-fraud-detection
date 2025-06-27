import pandas as pd
import json
import time
from kafka import KafkaProducer

df = pd.read_csv('../data/processed_transactions.csv')

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = 'transactions'
print("Starting the Kafka producer...")

for _, row in df.sample(50).iterrows():
    transaction = row.drop('Class').to_dict()
    producer.send(topic, transaction)
    print("Sent:", transaction)
    time.sleep(1)  # Simulate real-time data stream

producer.flush()