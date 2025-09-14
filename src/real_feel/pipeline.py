from real_feel.models import Tweet, init_db
from real_feel.twitter_client import TwitterClient
from real_feel.sentimentPredict import SentimentAnalyzer
from typing import List, Dict, Any
import json

class DataPipeline:
    def __init__(self, db_url: str, twitter_auth: Dict[str, str], rapidapi_key: str,
                 use_ensemble_bot_detector: bool = True, ensemble_model_path: str = None):
        """
        Initialize the data pipeline with CLS + MaxPool ensemble bot detection

        Args:
            db_url: Database connection URL
            twitter_auth: Twitter API credentials
            rapidapi_key: RapidAPI key for Botometer (fallback)
            use_ensemble_bot_detector: Whether to use transformer ensemble (default: True)
            ensemble_model_path: Path to trained ensemble model weights (optional)
        """
        self.session = init_db(db_url)
        self.twitter_client = TwitterClient(
            twitter_auth,
            rapidapi_key,
            use_ensemble_detector=use_ensemble_bot_detector,
            ensemble_model_path=ensemble_model_path
        )
        self.sentiment_analyzer = SentimentAnalyzer()

        print(f"✓ DataPipeline initialized with {'CLS+MaxPool ensemble' if use_ensemble_bot_detector else 'BotometerX'} bot detection")
    
    def process_tweets(self, query: str, max_tweets: int = 100) -> List[Tweet]:
        """
        Full pipeline to collect, analyze and store tweets
        
        Args:
            query: Search query string
            max_tweets: Maximum number of tweets to process
            
        Returns:
            List of processed Tweet objects
        """
        
        tweets = self.twitter_client.search_tweets(query, max_tweets)
        processed_tweets = []
        
        for tweet_data in tweets:
            # check if tweet exists
            existing = self.session.query(Tweet).filter_by(
                tweet_id = tweet_data['tweet_id']
            ).first()
            
            if existing:
                continue
                
            sentiment_result = self.sentiment_analyzer.sentimentAnalysis(tweet_data['text'])
            bot_result = self.twitter_client.check_bot(
                user_id=tweet_data['user_id'],
                tweet_text=tweet_data['text'],
                user_data=tweet_data.get('user', {})
            )
            
            tweet = Tweet(
                tweet_id                = tweet_data['tweet_id'],
                text                    = tweet_data['text'],
                user_id                 = tweet_data['user_id'],
                created_at              = tweet_data['created_at'],
                sentiment               = sentiment_result['predicted_sentiment'],
                sentiment_confidence    = sentiment_result['confidence'],
                sentiment_scores        = json.dumps(sentiment_result['sentiment_scores']),
                is_bot                  = bot_result['is_bot'],
                bot_score               = bot_result['bot_score'],
                bot_scores              = bot_result['bot_scores']
            )
            
            # save results to db
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
        
        # Get bot detection statistics
        bot_detection_stats = self.twitter_client.get_bot_detection_stats()

        return {
            'total_tweets': total_tweets,
            'bot_tweets': bot_tweets,
            'real_tweets': real_tweets,
            'sentiment_stats': sentiment_stats,
            'bot_detection_stats': bot_detection_stats
        }
