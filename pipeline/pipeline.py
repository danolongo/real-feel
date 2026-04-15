import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType, BooleanType, ArrayType

# --- Lazy singletons per worker process ---

_ort_session = None
_tokenizer = None
_sentiment_pipeline = None
_sentence_embedding = None

ONNX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/rf.v1.0.0.onnx"))
TOKENIZER_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
URL_RE = re.compile(r'http[s]?://\S+')
MENTION_RE = re.compile(r'@\w+')

def _get_bot_model():
    global _ort_session, _tokenizer
    if _ort_session is None:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        _ort_session = ort.InferenceSession(ONNX_PATH)
    return _ort_session, _tokenizer

def _get_sentiment_model():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(TOKENIZER_NAME)
        mdl.eval()
        _sentiment_pipeline = (mdl, tok)
    return _sentiment_pipeline

def _preprocess(text: str) -> str:
    text = URL_RE.sub('http://url.removed', text)
    text = MENTION_RE.sub('@user', text)
    return ' '.join(text.split())

@udf(returnType=FloatType())
def bot_probability(text: str) -> float:
    if not text:
        return 0.0
    session, tokenizer = _get_bot_model()
    encoded = tokenizer(
        _preprocess(text),
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
    return float(probs[0][1])  # P(bot)

@udf(returnType=StringType())
def sentiment(text: str) -> str:
    if not text:
        return "neutral"
    import torch
    import torch.nn.functional as F
    model, tokenizer = _get_sentiment_model()
    encoded = tokenizer(
        _preprocess(text),
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits
    probs = F.softmax(logits, dim=-1)[0]
    label_id = probs.argmax().item()
    return model.config.id2label[label_id].lower()

def _get_embedding_model():
    global _sentence_embedding
    if _sentence_embedding is None:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        mdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        mdl.eval()
        _sentence_embedding = (mdl, tok)
    return _sentence_embedding

@udf(returnType=ArrayType(FloatType()))
def embedding(text: str) -> list:
    if not text:
        return [0.0] * 384
    import torch
    mdl, tok = _get_embedding_model()
    encoded = tok(_preprocess(text), max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        hidden = mdl(**encoded).last_hidden_state  # [1, seq_len, 384]
    # mean pool over non-padding tokens
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled[0].numpy().astype(float).tolist()

def write_to_postgres(batch_df, batch_id):
    import psycopg2
    conn = psycopg2.connect(
        host="localhost", port=5433,
        dbname="realfeel", user="realfeel", password="realfeel"
    )
    cursor = conn.cursor()

    for row in batch_df.collect():
        cursor.execute(
            """
            INSERT INTO tweets (tweet_id, text, author, timestamp, bot_prob,
  is_bot, sentiment, embedding)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
              ON CONFLICT (tweet_id) DO NOTHING
            """, (row.tweet_id, row.text, row.author, row.timestamp,
                row.bot_prob, row.is_bot, row.sentiment, str(row.embedding))
        )

    conn.commit()
    cursor.close()
    conn.close()

# --- Spark session ---

spark = SparkSession.builder \
    .appName("real-feel-pipeline") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# --- Schema ---

tweet_schema = StructType([
    StructField("tweet_id", StringType()),
    StructField("text", StringType()),
    StructField("author", StringType()),
    StructField("timestamp", StringType()),
])

# --- Stream ---

raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "raw-tweets") \
    .option("startingOffsets", "earliest") \
    .load()

tweets = raw \
    .select(col("value").cast("string")) \
    .select(from_json(col("value"), tweet_schema).alias("t")) \
    .select("t.*") \
    .withColumn("bot_prob", bot_probability(col("text"))) \
    .withColumn("is_bot", col("bot_prob") > 0.5) \
    .withColumn("sentiment", sentiment(col("text"))) \
    .withColumn("embedding", embedding(col("text")))

query = tweets.writeStream \
    .foreachBatch(write_to_postgres) \
    .start()

query.awaitTermination()
