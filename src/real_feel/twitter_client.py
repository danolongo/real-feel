import tweepy
import botometer
from datetime import datetime
import json
from typing import List, Dict, Any

class TwitterClient:
    def __init__(self, twitter_auth: Dict[str, str], rapidapi_key: str):
        """
        Initialize Twitter and Botometer clients
        
        Args:
            twitter_auth: Dict containing Twitter API credentials
            rapidapi_key: RapidAPI key for Botometer
        """
        # Initialize Twitter client
        auth = tweepy.OAuthHandler(
            twitter_auth['consumer_key'],
            twitter_auth['consumer_secret']
        )
        auth.set_access_token(
            twitter_auth['access_token'],
            twitter_auth['access_token_secret']
        )
        self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
        
        # Initialize Botometer
        self.botometer = botometer.Botometer(
            wait_on_ratelimit=True,
            rapidapi_key=rapidapi_key,
            **twitter_auth
        )
    
    def search_tweets(self, query: str, max_tweets: int = 100) -> List[Dict[str, Any]]:
        """
        Search for tweets matching the query
        
        Args:
            query: Search query string
            max_tweets: Maximum number of tweets to retrieve
            
        Returns:
            List of tweet objects
        """
        tweets = []
        try:
            for tweet in tweepy.Cursor(
                self.twitter_api.search_tweets,
                q=query,
                lang="en",
                tweet_mode="extended"
            ).items(max_tweets):
                tweet_data = {
                    'tweet_id': str(tweet.id),
                    'text': tweet.full_text,
                    'user_id': str(tweet.user.id),
                    'created_at': tweet.created_at
                }
                tweets.append(tweet_data)
        except Exception as e:
            print(f"Error searching tweets: {e}")
            
        return tweets
    
    def check_bot(self, user_id: str) -> Dict[str, Any]:
        """
        Check if a user is likely to be a bot using Botometer
        
        Args:
            user_id: Twitter user ID
            
        Returns:
            Dictionary containing bot scores and classification
        """
        try:
            result = self.botometer.check_account(user_id)
            
            # Extract the English language scores
            scores = result.get('raw_scores', {}).get('english', {})
            overall_score = result.get('raw_scores', {}).get('universal', {})
            
            return {
                'is_bot': overall_score > 0.6,  # This threshold can be adjusted
                'bot_score': overall_score,
                'bot_scores': json.dumps(scores)
            }
        except Exception as e:
            print(f"Error checking bot status: {e}")
            return {
                'is_bot': None,
                'bot_score': None,
                'bot_scores': None
            }
