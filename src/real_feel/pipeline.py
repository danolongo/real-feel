from real_feel.models import Tweet, init_db
from real_feel.twitter_client import TwitterClient
from real_feel.sentimentPredict import SentimentAnalyzer
from typing import List, Dict, Any
import json
# from datetime import datetime

class DataPipeline:
    def __init__(self, db_url: str, twitter_auth: Dict[str, str], rapidapi_key: str):
        """
        Initialize the data pipeline
        
        Args:
            db_url: Database connection URL
            twitter_auth: Twitter API credentials
            rapidapi_key: RapidAPI key for Botometer
        """
        self.session = init_db(db_url)
        self.twitter_client = TwitterClient(twitter_auth, rapidapi_key)
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def process_tweets(self, query: str, max_tweets: int = 100) -> List[Tweet]:
        """
        Full pipeline to collect, analyze and store tweets
        
        Args:
            query: Search query string
            max_tweets: Maximum number of tweets to process
            
        Returns:
            List of processed Tweet objects
        """
        # Get tweets from Twitter
        tweets = self.twitter_client.search_tweets(query, max_tweets)
        processed_tweets = []
        
        for tweet_data in tweets:
            # Check if tweet already exists
            existing = self.session.query(Tweet).filter_by(
                tweet_id=tweet_data['tweet_id']
            ).first()
            
            if existing:
                continue
                
            # Run sentiment analysis
            sentiment_result = self.sentiment_analyzer.sentimentAnalysis(
                tweet_data['text']
            )
            
            # Check if user is a bot
            bot_result = self.twitter_client.check_bot(tweet_data['user_id'])
            
            # Create Tweet object
            tweet = Tweet(
                tweet_id=tweet_data['tweet_id'],
                text=tweet_data['text'],
                user_id=tweet_data['user_id'],
                created_at=tweet_data['created_at'],
                sentiment=sentiment_result['predicted_sentiment'],
                sentiment_confidence=sentiment_result['confidence'],
                sentiment_scores=json.dumps(sentiment_result['sentiment_scores']),
                is_bot=bot_result['is_bot'],
                bot_score=bot_result['bot_score'],
                bot_scores=bot_result['bot_scores']
            )
            
            # Save to database
            try:
                self.session.add(tweet)
                self.session.commit()
                processed_tweets.append(tweet)
            except Exception as e:
                print(f"Error saving tweet: {e}")
                self.session.rollback()
        
        return processed_tweets
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about processed tweets
        
        Returns:
            Dictionary containing various statistics
        """
        total_tweets = self.session.query(Tweet).count()
        bot_tweets = self.session.query(Tweet).filter_by(is_bot=True).count()
        real_tweets = total_tweets - bot_tweets
        
        sentiment_stats = {
            'real': {
                'positive': self.session.query(Tweet).filter_by(
                    is_bot=False, sentiment='positive'
                ).count(),
                'neutral': self.session.query(Tweet).filter_by(
                    is_bot=False, sentiment='neutral'
                ).count(),
                'negative': self.session.query(Tweet).filter_by(
                    is_bot=False, sentiment='negative'
                ).count()
            },
            'bot': {
                'positive': self.session.query(Tweet).filter_by(
                    is_bot=True, sentiment='positive'
                ).count(),
                'neutral': self.session.query(Tweet).filter_by(
                    is_bot=True, sentiment='neutral'
                ).count(),
                'negative': self.session.query(Tweet).filter_by(
                    is_bot=True, sentiment='negative'
                ).count()
            }
        }
        
        return {
            'total_tweets': total_tweets,
            'bot_tweets': bot_tweets,
            'real_tweets': real_tweets,
            'sentiment_stats': sentiment_stats
        }
