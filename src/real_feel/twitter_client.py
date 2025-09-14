import tweepy
import botometer
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

        self.twitter_client = tweepy.Client(
            bearer_token = twitter_auth['bearer_token'],
            consumer_key = twitter_auth['consumer_key'],
            consumer_secret = twitter_auth['consumer_secret'],
            access_token = twitter_auth['access_token'],
            access_token_secret = twitter_auth['access_token_secret'],
            wait_on_rate_limit = True
        )
        
        self.botometer = botometer.BotometerX(
            wait_on_ratelimit = True,
            rapidapi_key = rapidapi_key,
            **twitter_auth
        )
    
    def search_tweets(self, query: str, max_tweets: int = 100) -> List[Dict[str, Any]]:
        """
        Search for tweets matching the query using Twitter API v2
        
        Args:
            query: Search query string
            max_tweets: Maximum number of tweets to retrieve
            
        Returns:
            List of tweet objects
        """

        tweets = []

        try:
            response = self.twitter_client.search_recent_tweets(
                query = query,
                max_results = min(max_tweets, 100),  # API v2 has a limit of 100 tweets per request
                tweet_fields = ['created_at', 'text', 'author_id'],
                user_fields = ['id', 'username'],
                expansions = ['author_id']
            )

            # add tweet info to tweet_data
            for tweet in response.data:
                tweet_data = {
                    'tweet_id': str(tweet.id),
                    'text': tweet.text,
                    'user_id': str(tweet.author_id),
                    'created_at': tweet.created_at
                }
                
                tweets.append(tweet_data)
            
            # if we need more tweets, use pagination
            if max_tweets > 100 and response.meta.get('next_token'):
                while len(tweets) < max_tweets and response.meta.get('next_token'):
                    response = self.twitter_client.search_recent_tweets(
                        query = query,
                        max_results = 100,
                        tweet_fields = ['created_at', 'text', 'author_id'],
                        user_fields = ['id', 'username'],
                        expansions = ['author_id'],
                        next_token = response.meta['next_token']
                    )
                    
                    for tweet in response.data:
                        if len(tweets) >= max_tweets:
                            break
                        tweet_data = {
                            'tweet_id': str(tweet.id),
                            'text': tweet.text,
                            'user_id': str(tweet.author_id),
                            'created_at': tweet.created_at
                        }
                        tweets.append(tweet_data)
                        
        except Exception as e:
            print(f"Error searching tweets: {e}")
            
        return tweets
    

    """
    ALERT! once bot detector model is usable, change this code
    """
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
            
            # extract the english language scores
            scores = result.get('raw_scores', {}).get('english', {})
            overall_score = result.get('raw_scores', {}).get('universal', {})

            threshold = 0.6 # adjustable

            return {
                'is_bot': overall_score > threshold,  
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
