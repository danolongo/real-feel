import os
import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

from .dataProcessing import DataProcessor

class SentimentAnalyzer:
    def __init__(self, modelPath=None):
        self.dataProcessor = DataProcessor()
        self.model = None
        self.tokenizer = None
        self.labels = []
        
        if modelPath is None:
            modelPath = self.getDefaultModelPath()
        
        self.modelPath = modelPath
        self.loadModel()
    
    def getDefaultModelPath(self):
        currentDir = Path(__file__).parent
        modelDir = currentDir / "models" / "twitter-roberta-base-sentiment"
        return str(modelDir)
    
    def loadModel(self):
        try:
            if os.path.exists(self.modelPath):
                print(f"Loading local model from {self.modelPath}")
                self.model = AutoModelForSequenceClassification.from_pretrained(self.modelPath)
                
                tokenizerPath = self.modelPath.replace("sentiment", "tokenizer")
                if os.path.exists(tokenizerPath):
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizerPath, use_fast=True)
                else:
                    # if local does not exist go back to online tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", use_fast=True)
            else:
                # dowload online model if it doesnt exist locally
                print("Local model not found, downloading...")
                MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
                self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
                
                # save locally for future use
                os.makedirs(os.path.dirname(self.modelPath), exist_ok=True)
                self.model.save_pretrained(self.modelPath)
                
                tokenizerPath = self.modelPath.replace("sentiment", "tokenizer") 
                self.tokenizer.save_pretrained(tokenizerPath)
            
            self.labels = ['negative', 'neutral', 'positive']
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    #########################################################################################################
    #########################################################################################################
    # THIS FUNCTION SHOULD BE USED IN LINE 56 IF USING MORE THAN THE SENTIMENT MODEL
    # whenever more than one task from model is being used, multiple labels should be generated
    # def _load_labels(self):
    #     """Download and parse sentiment labels"""
    #     try:
    #         mapping_link = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
    #         with urllib.request.urlopen(mapping_link) as f:
    #             html = f.read().decode('utf-8').split("\n")
    #             csvreader = csv.reader(html, delimiter='\t')
    #             self.labels = [row[1] for row in csvreader if len(row) > 1]
    #     except:
    #         # Fallback labels if download fails
    #         self.labels = ['negative', 'neutral', 'positive']
    #########################################################################################################
    #########################################################################################################
    
    def sentimentAnalysis(self, text):
        """        
        Args:
            text (str): Tweet text to analyze
            
        Returns:
            dict: Dictionary with sentiment scores and prediction
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")
        
        processedText = self.dataProcessor.preprocess(text)
        
        # tokenize and predict
        encodedInput = self.tokenizer(processedText, return_tensors='pt', max_length=512, truncation=True)
        output = self.model(**encodedInput)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        result = {
            'text': text,
            'processed_text': processedText,
            'sentiment_scores': {},
            'predicted_sentiment': '',
            'confidence': float(np.max(scores))
        }
        
        for i, label in enumerate(self.labels):
            result['sentiment_scores'][label] = float(scores[i])
        
        predictedIdx = np.argmax(scores)
        result['predicted_sentiment'] = self.labels[predictedIdx]
        
        return result
    
    def analyzeBatch(self, texts):
        """
        Args:
            texts (list): List of tweet texts
            
        Returns:
            list: List of sentiment analysis results
        """
        return [self.sentimentAnalysis(text) for text in texts]
