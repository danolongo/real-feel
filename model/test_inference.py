"""
Quick sanity check for the exported ONNX model.
Run from repo root: pipeline/.venv/bin/python model/test_inference.py
"""

import re
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ONNX_PATH = "model/rf.v1.0.0.onnx"
TOKENIZER = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
session = ort.InferenceSession(ONNX_PATH)

URL_PATTERN = re.compile(r'http[s]?://\S+')
MENTION_PATTERN = re.compile(r'@\w+')

def preprocess(text: str) -> str:
    text = URL_PATTERN.sub('http://url.removed', text)
    text = MENTION_PATTERN.sub('@user', text)
    return ' '.join(text.split())

def predict(text: str) -> dict:
    text = preprocess(text)
    encoded = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    logits = session.run(None, {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    })[0]

    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    predicted_class = int(np.argmax(probs))

    return {
        "text": text,
        "prediction": "bot" if predicted_class == 1 else "human",
        "prob_human": float(probs[0][0]),
        "prob_bot": float(probs[0][1]),
    }

tweets = [
    # raw tweet text
    "Just had the best coffee this morning, feeling ready to tackle the day!",
    "CLICK HERE FREE IPHONE WINNER CLAIM NOW LIMITED TIME OFFER BUY CRYPTO",
    # user profile format (how training data was structured)
    "Name: Daniel Martinez | Username: @danolongo | Bio: software engineer, coffee lover, hiking on weekends",
    "Name: xX_crypto_Xx | Username: @user | Bio: Follow for follow! DM for promo! Get rich fast! http://url.removed",
    "Name: bot29472910 | Username: @user | Bio: ",
]

for tweet in tweets:
    result = predict(tweet)
    print(f"\nText:    {result['text']}")
    print(f"Result:  {result['prediction'].upper()}")
    print(f"P(human): {result['prob_human']:.4f}  P(bot): {result['prob_bot']:.4f}")
