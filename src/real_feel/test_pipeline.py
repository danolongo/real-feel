"""
Test script for the Twitter sentiment and bot detection pipeline.
Before running this script:
1. Update config.py with your credentials
2. Ensure PostgreSQL is running
3. Create the database: createdb realFeelDB
"""

from real_feel.pipeline import DataPipeline
from real_feel.config import DATABASE_URL, TWITTER_AUTH, RAPIDAPI_KEY
import json

def test_pipeline():
    # Initialize pipeline
    pipeline = DataPipeline(
        db_url=DATABASE_URL,
        twitter_auth=TWITTER_AUTH,
        rapidapi_key=RAPIDAPI_KEY
    )
    
    query = input("enter query: ")

    # Test tweet collection and analysis
    print("Processing tweets...")
    tweets = pipeline.process_tweets(
        query,
        max_tweets=10  # Small number for testing
    )
    
    print(f"\nProcessed {len(tweets)} tweets")
    
    # Get and display statistics
    print("\nGetting statistics...")
    stats = pipeline.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Display some example tweets with their analysis
    print("\nExample processed tweets:")
    for tweet in tweets[:3]:  # Show first 3 tweets
        print("\nTweet ID:", tweet.tweet_id)
        print("Text:", tweet.text[:100] + "..." if len(tweet.text) > 100 else tweet.text)
        print("Sentiment:", tweet.sentiment)
        print("Bot Score:", tweet.bot_score)
        print("Is Bot:", tweet.is_bot)

if __name__ == "__main__":
    test_pipeline()
