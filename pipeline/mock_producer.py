"""
simply tests that kafka producer is sending the messages
with some mock tweets
"""

from kafka import KafkaProducer
import json
from datetime import datetime
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


example = [
    {
        "tweet_id": "123e4567-e89b-12d3-a456-426614174000",
        "text": "hi",
        "author": "juana",
        "timestamp": datetime.now().isoformat()
    },
    {
        "tweet_id": "123e4567-e89b-12d3-a456-426614174001",
        "text": "hi",
        "author": "juana",
        "timestamp": datetime.now().isoformat()
    },
    {
        "tweet_id": "123e4567-e89b-12d3-a456-426614174002",
        "text": "hi",
        "author": "juana",
        "timestamp": datetime.now().isoformat()
    },
    {
        "tweet_id": "123e4567-e89b-12d3-a456-426614174003",
        "text": "hi",
        "author": "juana",
        "timestamp": datetime.now().isoformat()
    },
    {
        "tweet_id": "123e4567-e89b-12d3-a456-426614174004",
        "text": "hi",
        "author": "juana",
        "timestamp": datetime.now().isoformat()
    },
]

topic = "raw-tweets"

for i, tweet in enumerate(example, start=1):
    producer.send(topic, value=tweet)
    print(f"[producer] sent [{i:02d}/{len(example)}]: '{tweet}'")
    time.sleep(1.5)

producer.flush()

