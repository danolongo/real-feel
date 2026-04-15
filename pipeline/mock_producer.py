"""
simply tests that kafka producer is sending the messages
with some mock tweets
"""

from kafka import KafkaProducer
import json
import uuid
from datetime import datetime
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

example = [
    {"tweet_id": str(uuid.uuid4()), "text": "I absolutely love this, best day ever!", "author": "alice", "timestamp": datetime.now().isoformat()},
    {"tweet_id": str(uuid.uuid4()), "text": "This is terrible, completely broken and awful", "author": "bob", "timestamp": datetime.now().isoformat()},
    {"tweet_id": str(uuid.uuid4()), "text": "The meeting is at 3pm tomorrow", "author": "carol", "timestamp": datetime.now().isoformat()},
    {"tweet_id": str(uuid.uuid4()), "text": "Just woke up, going to make coffee", "author": "dave", "timestamp": datetime.now().isoformat()},
    {"tweet_id": str(uuid.uuid4()), "text": "Absolutely disgusted by this behavior, unacceptable!", "author": "eve", "timestamp": datetime.now().isoformat()},
]

topic = "raw-tweets"

for i, tweet in enumerate(example, start=1):
    producer.send(topic, value=tweet)
    print(f"[producer] sent [{i:02d}/{len(example)}]: '{tweet}'")
    time.sleep(1.5)

producer.flush()
