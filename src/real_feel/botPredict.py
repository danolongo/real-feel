import os
import joblib

class BotDetector:
    def __init__(self, modelPath=None):
        self.model = None
        self.featureExtractors = self.initFeatureExtractors()
        
        if modelPath and os.path.exists(modelPath):
            self.loadModel(modelPath)
    
    def initFeatureExtractors(self):
        return {
            'text_features': self.extractTextFeatures,
            'user_features': self.extractUserFeatures,
            'temporal_features': self.extractTemporalFeatures
        }
    
    def extractTextFeatures(self, tweet):
        """        
        Args:
            tweet (dict): Tweet data dictionary
            
        Returns:
            dict: Text-based features
        """
        text = tweet.get('text', '')
        
        features = {
            'text_length': len(text),
            'url_count': text.count('http'),
            'mention_count': text.count('@'),
            'hashtag_count': text.count('#'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
        }
        
        return features
    
    def extractUserFeatures(self, tweet):
        """        
        Args:
            tweet (dict): Tweet data dictionary
            
        Returns:
            dict: User-based features
        """
        user = tweet.get('user', {})
        
        features = {
            'followers_count': user.get('followers_count', 0),
            'friends_count': user.get('friends_count', 0),
            'statuses_count': user.get('statuses_count', 0),
            'favourites_count': user.get('favourites_count', 0),
            'listed_count': user.get('listed_count', 0),
            'verified': int(user.get('verified', False)),
            'default_profile': int(user.get('default_profile', True)),
            'default_profile_image': int(user.get('default_profile_image', True)),
            'has_description': int(bool(user.get('description', ''))),
            'name_length': len(user.get('name', '')),
            'screen_name_length': len(user.get('screen_name', '')),
        }
        
        followers = max(features['followers_count'], 1)
        friends = max(features['friends_count'], 1)
        
        features.update({
            'followers_friends_ratio': followers / friends,
            'friends_followers_ratio': friends / followers,
            'statuses_per_day': features['statuses_count'] / max(self.getAccountAgeDays(user), 1)
        })
        
        return features
    
    def extractTemporalFeatures(self, tweet):
        """        
        Args:
            tweet (dict): Tweet data dictionary
            
        Returns:
            dict: Temporal features
        """
        # This would analyze posting patterns, timing, etc.
        # Simplified version for now
        return {
            'hour_of_day': 12,  # Would extract from created_at
            'day_of_week': 1,   # Would extract from created_at
        }
    
    def getAccountAgeDays(self, user):
        """        
        Args:
            user (dict): User data dictionary
            
        Returns:
            int: Account age in days
        """
        # simplified - would parse created_at timestamp
        return max(user.get('statuses_count', 0) // 10, 1)  # rough estimate
    
    def isItBot(self, tweet):
        """        
        Args:
            tweet (dict): Tweet data dictionary
            
        Returns:
            dict: Dictionary with bot prediction and confidence
        """
        allFeatures = {}
        for extractorName, extractorFunc in self.featureExtractors.items():
            features = extractorFunc(tweet)
            allFeatures.update(features)
        
        # Simple rule-based detection for now
        # In a real implementation, this would use a trained ML model
        botScore = self.calculateBotScore(allFeatures)
        isBot = botScore > 0.5
        
        result = {
            'tweet_id': tweet.get('id', 'unknown'),
            'is_bot': isBot,
            'bot_confidence': botScore,
            'features': allFeatures,
            'detection_method': 'rule_based'  # or 'ml_model' when trained
        }
        
        return result
    
    def calculateBotScore(self, features):
        """
        Calculate bot probability score based on features
        
        Args:
            features (dict): Extracted features
            
        Returns:
            float: Bot probability score (0-1)
        """
        score = 0.0
        
        # Rule-based scoring (simplified)
        if features.get('followers_count', 0) < 10:
            score += 0.2
        
        if features.get('default_profile_image', 0) == 1:
            score += 0.15
        
        if features.get('default_profile', 0) == 1:
            score += 0.1
        
        if features.get('followers_friends_ratio', 1) < 0.1:
            score += 0.2
        
        if features.get('caps_ratio', 0) > 0.3:
            score += 0.1
        
        if features.get('url_count', 0) > 2:
            score += 0.15
        
        if not features.get('has_description', 0):
            score += 0.1
        
        return min(score, 1.0)
    
    def detectBatch(self, tweets):
        """        
        Args:
            tweets (list): List of tweet dictionaries
            
        Returns:
            list: List of bot detection results
        """
        return [self.isItBot(tweet) for tweet in tweets]
    
    def loadModel(self, modelPath):
        """Load a trained bot detection model"""
        self.model = joblib.load(modelPath)
    
    def saveModel(self, modelPath):
        """Save the current bot detection model"""
        if self.model:
            joblib.dump(self.model, modelPath)
