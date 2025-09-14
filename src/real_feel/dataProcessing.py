import re
import pandas as pd

class DataProcessor:
    def __init__(self):
        self.urlPattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mentionPattern = re.compile(r'@\w+')
        
    def preprocess(self, text):
        """
        Preprocess tweet text by removing URLs and normalizing mentions.
        
        Args:
            text: Raw tweet text
            
        Returns:
            Preprocessed text with URLs and mentions normalized
        """
        text = self.urlPattern.sub('http://url.removed', text)
        text = self.mentionPattern.sub('@user', text)
        text = ' '.join(text.split())
        
        return text
    
    def processBatch(self, tweets):
        """
        Process a batch of tweets into a pandas DataFrame.
        
        Args:
            tweets: List of tweet dictionaries
            
        Returns:
            DataFrame containing processed tweets with processed_text column
        """
        df = pd.DataFrame(tweets)
        if 'text' in df.columns:
            df['processed_text'] = df['text'].apply(self.preprocess)
        
        return df
    
    def amounts(self, predictions):
        """
        Calculate counts of bot vs real tweets from predictions.
        
        Args:
            predictions: List of prediction dictionaries containing 'is_bot' key
        
        Returns:
            Tuple containing (bot_count, real_count, total_count)
        """
        botTweets = sum(1 for pred in predictions if pred.get('is_bot', False))
        realTweets = len(predictions) - botTweets
        totalTweets = len(predictions)
        
        return botTweets, realTweets, totalTweets
    
    def filterBots(self, tweets, predictions, threshold=0.5):
        """
        Filter out bot tweets based on predictions and confidence threshold.
        
        Args:
            tweets: List of tweet dictionaries
            predictions: List of bot predictions
            threshold: Confidence threshold for filtering (default: 0.5)
            
        Returns:
            List of filtered tweets without detected bots
        """
        filtered_tweets = []
        
        for tweet, pred in zip(tweets, predictions):
            if not pred.get('is_bot', False) or pred.get('bot_confidence', 0) < threshold:
                filtered_tweets.append(tweet)

        return filtered_tweets