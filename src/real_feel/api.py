from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json

from .pipeline import DataPipeline
from .config import DATABASE_URL, TWITTER_AUTH, RAPIDAPI_KEY
from .models import Tweet

app = FastAPI(
    title="RealFeel API",
    description="API for Twitter sentiment analysis and bot detection",
    version="1.0.0"
)

# Initialize pipeline
pipeline = DataPipeline(
    db_url=DATABASE_URL,
    twitter_auth=TWITTER_AUTH,
    rapidapi_key=RAPIDAPI_KEY
)

class TweetResponse(BaseModel):
    tweet_id: str
    text: str
    sentiment: str
    sentiment_confidence: float
    is_bot: bool
    bot_score: float

class StatsResponse(BaseModel):
    total_tweets: int
    bot_tweets: int
    real_tweets: int
    sentiment_stats: dict

@app.get("/")
async def root():
    return {"message": "Welcome to RealFeel API"}

@app.post("/analyze", response_model=List[TweetResponse])
async def analyze_tweets(query: str, max_tweets: Optional[int] = 100):
    """
    Analyze tweets for a given query
    
    Args:
        query: Search query string
        max_tweets: Maximum number of tweets to analyze (default: 100)
    """
    try:
        tweets = pipeline.process_tweets(query, max_tweets)
        return [{
            "tweet_id": tweet.tweet_id,
            "text": tweet.text,
            "sentiment": tweet.sentiment,
            "sentiment_confidence": tweet.sentiment_confidence,
            "is_bot": tweet.is_bot,
            "bot_score": tweet.bot_score
        } for tweet in tweets]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get statistics about analyzed tweets"""
    try:
        return pipeline.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tweets/{sentiment_type}")
async def get_tweets_by_sentiment(
    sentiment_type: str,
    bot_only: Optional[bool] = False,
    limit: Optional[int] = 10
):
    """
    Get tweets by sentiment type
    
    Args:
        sentiment_type: One of 'positive', 'neutral', 'negative'
        bot_only: If True, return only bot tweets
        limit: Maximum number of tweets to return
    """
    if sentiment_type not in ['positive', 'neutral', 'negative']:
        raise HTTPException(
            status_code=400,
            detail="Invalid sentiment type. Must be one of: positive, neutral, negative"
        )
    
    try:
        query = pipeline.session.query(Tweet).filter_by(sentiment=sentiment_type)
        if bot_only is not None:
            query = query.filter_by(is_bot=bot_only)
        
        tweets = query.limit(limit).all()
        return [{
            "tweet_id": tweet.tweet_id,
            "text": tweet.text,
            "sentiment": tweet.sentiment,
            "sentiment_confidence": tweet.sentiment_confidence,
            "is_bot": tweet.is_bot,
            "bot_score": tweet.bot_score
        } for tweet in tweets]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
