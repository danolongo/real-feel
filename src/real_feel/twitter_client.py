import tweepy
import botometer
import json
from typing import List, Dict, Any, Optional
from real_feel.ensemble_bot_detector import create_ensemble_bot_detector

class TwitterClient:
    def __init__(self, twitter_auth: Dict[str, str], rapidapi_key: str,
                 use_ensemble_detector: bool = True, ensemble_model_path: Optional[str] = None):
        """
        Initialize Twitter client with ensemble bot detection

        Args:
            twitter_auth: Dict containing Twitter API credentials
            rapidapi_key: RapidAPI key for Botometer (fallback)
            use_ensemble_detector: Whether to use CLS+MaxPool ensemble (default: True)
            ensemble_model_path: Path to trained ensemble model (optional)
        """

        self.twitter_client = tweepy.Client(
            bearer_token = twitter_auth['bearer_token'],
            consumer_key = twitter_auth['consumer_key'],
            consumer_secret = twitter_auth['consumer_secret'],
            access_token = twitter_auth['access_token'],
            access_token_secret = twitter_auth['access_token_secret'],
            wait_on_rate_limit = True
        )

        # Initialize bot detection system
        self.use_ensemble_detector = use_ensemble_detector

        if use_ensemble_detector:
            # Use our CLS + MaxPool ensemble detector
            self.ensemble_detector = create_ensemble_bot_detector(
                model_path=ensemble_model_path
            )
            print("✓ Initialized CLS + MaxPool ensemble bot detector")
        else:
            # Fallback to BotometerX
            self.botometer = botometer.BotometerX(
                wait_on_ratelimit = True,
                rapidapi_key = rapidapi_key,
                **twitter_auth
            )
            print("✓ Initialized BotometerX detector (fallback)")
    
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
    

    def check_bot(self, user_id: str, tweet_text: str = "", user_data: Dict = None) -> Dict[str, Any]:
        """
        Check if a user is likely to be a bot using CLS + MaxPool ensemble or BotometerX fallback

        Args:
            user_id: Twitter user ID
            tweet_text: Tweet text for ensemble analysis
            user_data: Additional user metadata for analysis

        Returns:
            Dictionary containing bot scores and classification
        """
        if self.use_ensemble_detector:
            # Use CLS + MaxPool ensemble detector
            return self.ensemble_detector.check_bot(
                user_id=user_id,
                tweet_text=tweet_text,
                user_data=user_data
            )
        else:
            # Fallback to BotometerX
            try:
                result = self.botometer.check_account(user_id)

                # Extract the english language scores
                scores = result.get('raw_scores', {}).get('english', {})
                overall_score = result.get('raw_scores', {}).get('universal', {})

                threshold = 0.6  # adjustable

                return {
                    'is_bot': overall_score > threshold,
                    'bot_score': overall_score,
                    'bot_scores': json.dumps(scores)
                }
            except Exception as e:
                print(f"Error checking bot status with BotometerX: {e}")
                return {
                    'is_bot': None,
                    'bot_score': None,
                    'bot_scores': None
                }

    def check_bot_batch(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch bot detection for multiple tweets

        Args:
            tweets: List of tweet dictionaries with 'user_id', 'text', and optionally 'user'

        Returns:
            List of bot detection results
        """
        if self.use_ensemble_detector:
            return self.ensemble_detector.detect_batch(tweets)
        else:
            # BotometerX doesn't support efficient batching
            results = []
            for tweet in tweets:
                result = self.check_bot(
                    user_id=tweet.get('user_id', ''),
                    tweet_text=tweet.get('text', ''),
                    user_data=tweet.get('user', {})
                )
                results.append(result)
            return results

    def get_bot_detection_stats(self) -> Dict[str, Any]:
        """
        Get bot detection performance statistics

        Returns:
            Dictionary with detection statistics
        """
        if self.use_ensemble_detector:
            return self.ensemble_detector.get_detection_statistics()
        else:
            return {
                'detector_type': 'botometer_x',
                'message': 'BotometerX does not provide detailed statistics'
            }
