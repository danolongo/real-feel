import re
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        self.urlPattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mentionPattern = re.compile(r'@\w+')
        
    def preprocess(self, text):
        """
        Args:
            text (str): Raw tweet text
            
        Returns:
            str: Preprocessed text
        """
        text = self.urlPattern.sub('http://url.removed', text)
        text = self.mentionPattern.sub('@user', text)
        text = ' '.join(text.split())
        
        return text
    
    def processBatch(self, tweets):
        """        
        Args:
            tweets (list): List of tweet dictionaries
            
        Returns:
            pandas.DataFrame: Processed tweets
        """
        df = pd.DataFrame(tweets)
        if 'text' in df.columns:
            df['processed_text'] = df['text'].apply(self.preprocess)
        
        return df
    
    def amounts(self, predictions):
        """        
        Args:
            predictions (list): List of prediction dictionaries containing 'is_bot' key
        
        Returns:
            tuple: (bot_count, real_count, total_count)
        """
        botTweets = sum(1 for pred in predictions if pred.get('is_bot', False))
        realTweets = len(predictions) - botTweets
        totalTweets = len(predictions)
        
        return botTweets, realTweets, totalTweets
    
    def filterBots(self, tweets, predictions, threshold=0.5):
        """        
        Args:
            tweets (list): List of tweet dictionaries
            predictions (list): List of bot predictions
            threshold (float): Confidence threshold for filtering
            
        Returns:
            list: Filtered tweets without bots
        """
        filtered_tweets = []
        for tweet, pred in zip(tweets, predictions):
            if not pred.get('is_bot', False) or pred.get('bot_confidence', 0) < threshold:
                filtered_tweets.append(tweet)
        return filtered_tweets