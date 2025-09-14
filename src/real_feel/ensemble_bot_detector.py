"""
CLS + MaxPool Ensemble Bot Detection for RealFeel Pipeline
Replaces the rule-based BotometerX system with transformer-based detection
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, List, Any, Optional, Union
import json
import logging
from pathlib import Path

# Import our ensemble model
from real_feel.textModel.real_feel_model import (
    create_ensemble_model,
    get_production_config
)

logger = logging.getLogger(__name__)


class EnsembleBotDetector:
    """
    Production-ready CLS + MaxPool ensemble bot detector
    Integrates seamlessly with existing RealFeel pipeline
    """

    def __init__(self, model_path: Optional[str] = None, config_override: Optional[Dict] = None):
        """
        Initialize the ensemble bot detector

        Args:
            model_path: Path to trained model weights (optional)
            config_override: Configuration overrides (optional)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = get_production_config()

        # Initialize tokenizer (using RoBERTa as default)
        model_name = (config_override or {}).get('model_name', 'cardiffnlp/twitter-roberta-base-sentiment-latest')

        # Apply configuration overrides
        if config_override:
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize model
        self.model = create_ensemble_model(self.config.model, self.config.ensemble)
        self.model.to(self.device)
        self.model.eval()

        # Load trained weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            logger.info(f"Loaded ensemble model from {model_path}")
        else:
            logger.warning("No trained model loaded - using randomly initialized weights")

        # Performance tracking
        self.prediction_stats = {
            'total_predictions': 0,
            'bot_predictions': 0,
            'human_predictions': 0,
            'high_confidence_predictions': 0,
            'low_agreement_predictions': 0
        }

    def load_model(self, model_path: str):
        """Load trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise

    def save_model(self, model_path: str):
        """Save current model weights"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'stats': self.prediction_stats
            }, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess text for the ensemble model

        Args:
            text: Input text to process

        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        # Clean and truncate text
        text = str(text).strip()
        if not text:
            text = "[EMPTY]"

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.model.max_seq_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }

    def extract_user_features(self, tweet_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract interpretable user features for analysis
        Compatible with existing pipeline expectations

        Args:
            tweet_data: Tweet data dictionary

        Returns:
            Dictionary of user features
        """
        text = tweet_data.get('text', '')
        user_data = tweet_data.get('user', {})

        features = {
            # Text features
            'text_length': len(text),
            'url_count': text.count('http'),
            'mention_count': text.count('@'),
            'hashtag_count': text.count('#'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),

            # User features (if available)
            'followers_count': user_data.get('followers_count', 0),
            'friends_count': user_data.get('friends_count', 0),
            'statuses_count': user_data.get('statuses_count', 0),
            'verified': int(user_data.get('verified', False)),
            'default_profile': int(user_data.get('default_profile', True)),
            'has_description': int(bool(user_data.get('description', '')))
        }

        return features

    def check_bot(self, user_id: str, tweet_text: str = "", user_data: Dict = None) -> Dict[str, Any]:
        """
        Main bot detection method - compatible with existing pipeline interface

        Args:
            user_id: Twitter user ID
            tweet_text: Tweet text for analysis
            user_data: Additional user metadata (optional)

        Returns:
            Dictionary containing bot prediction results (compatible with pipeline)
        """
        try:
            # Prepare input for ensemble model
            if not tweet_text.strip():
                tweet_text = f"User profile analysis for {user_id}"

            # Preprocess text
            inputs = self.preprocess_text(tweet_text)

            # Get ensemble predictions
            with torch.no_grad():
                reasoning = self.model.predict_with_reasoning(
                    inputs['input_ids'],
                    inputs['attention_mask']
                )

            # Extract predictions (convert to scalar values)
            ensemble_pred = reasoning['predictions'][0].item()
            primary_confidence = reasoning['primary_confidence'][0].item()
            backup_confidence = reasoning['backup_confidence'][0].item()
            agreement = reasoning['agreement'][0].item()

            # Calculate overall confidence and bot score
            is_bot = ensemble_pred == 1
            bot_score = max(primary_confidence, backup_confidence) if is_bot else (1 - max(primary_confidence, backup_confidence))

            # Update statistics
            self._update_stats(is_bot, max(primary_confidence, backup_confidence), agreement)

            # Extract traditional features for compatibility
            tweet_data = {
                'text': tweet_text,
                'user': user_data or {},
                'id': user_id
            }
            traditional_features = self.extract_user_features(tweet_data)

            # Create detailed result compatible with existing pipeline
            result = {
                # Core prediction (pipeline compatibility)
                'is_bot': is_bot,
                'bot_score': bot_score,
                'bot_scores': json.dumps({
                    'ensemble_confidence': max(primary_confidence, backup_confidence),
                    'primary_cls_confidence': primary_confidence,
                    'backup_maxpool_confidence': backup_confidence,
                    'model_agreement': agreement,
                    'detection_method': 'cls_maxpool_ensemble'
                }),

                # Extended ensemble information
                'ensemble_details': {
                    'primary_prediction': 'bot' if reasoning['primary_predictions'][0].item() == 1 else 'human',
                    'backup_prediction': 'bot' if reasoning['backup_predictions'][0].item() == 1 else 'human',
                    'ensemble_prediction': 'bot' if is_bot else 'human',
                    'primary_confidence': primary_confidence,
                    'backup_confidence': backup_confidence,
                    'model_agreement': agreement,
                    'confidence_level': self._get_confidence_level(max(primary_confidence, backup_confidence))
                },

                # Traditional features for analysis
                'features': traditional_features
            }

            logger.debug(f"Bot detection for {user_id}: {result['ensemble_details']['ensemble_prediction']} "
                        f"(confidence: {bot_score:.3f}, agreement: {agreement:.3f})")

            return result

        except Exception as e:
            logger.error(f"Error in ensemble bot detection for {user_id}: {e}")

            # Return fallback result
            return {
                'is_bot': None,
                'bot_score': None,
                'bot_scores': json.dumps({
                    'error': str(e),
                    'detection_method': 'cls_maxpool_ensemble_failed'
                }),
                'ensemble_details': {
                    'error': str(e)
                },
                'features': {}
            }

    def detect_batch(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch bot detection for multiple tweets

        Args:
            tweets: List of tweet dictionaries

        Returns:
            List of bot detection results
        """
        results = []

        try:
            # Extract texts and prepare batch
            texts = []
            for tweet in tweets:
                text = tweet.get('text', f"Profile analysis for {tweet.get('user_id', 'unknown')}")
                texts.append(text)

            # Batch tokenization
            batch_encoding = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.config.model.max_seq_length,
                return_tensors='pt'
            )

            batch_inputs = {
                'input_ids': batch_encoding['input_ids'].to(self.device),
                'attention_mask': batch_encoding['attention_mask'].to(self.device)
            }

            # Batch prediction
            with torch.no_grad():
                reasoning = self.model.predict_with_reasoning(
                    batch_inputs['input_ids'],
                    batch_inputs['attention_mask']
                )

            # Process results
            for i, tweet in enumerate(tweets):
                user_id = tweet.get('user_id', 'unknown')

                # Extract individual predictions
                ensemble_pred = reasoning['predictions'][i].item()
                primary_confidence = reasoning['primary_confidence'][i].item()
                backup_confidence = reasoning['backup_confidence'][i].item()
                agreement = reasoning['agreement'][i].item()

                is_bot = ensemble_pred == 1
                bot_score = max(primary_confidence, backup_confidence) if is_bot else (1 - max(primary_confidence, backup_confidence))

                # Update statistics
                self._update_stats(is_bot, max(primary_confidence, backup_confidence), agreement)

                # Create result
                result = self.check_bot(user_id, texts[i], tweet.get('user', {}))
                results.append(result)

        except Exception as e:
            logger.error(f"Error in batch bot detection: {e}")
            # Return fallback results for all tweets
            for tweet in tweets:
                results.append({
                    'is_bot': None,
                    'bot_score': None,
                    'bot_scores': json.dumps({'error': str(e)}),
                    'ensemble_details': {'error': str(e)},
                    'features': {}
                })

        return results

    def _update_stats(self, is_bot: bool, confidence: float, agreement: float):
        """Update internal prediction statistics"""
        self.prediction_stats['total_predictions'] += 1

        if is_bot:
            self.prediction_stats['bot_predictions'] += 1
        else:
            self.prediction_stats['human_predictions'] += 1

        if confidence > 0.8:
            self.prediction_stats['high_confidence_predictions'] += 1

        if agreement < 0.5:
            self.prediction_stats['low_agreement_predictions'] += 1

    def _get_confidence_level(self, confidence: float) -> str:
        """Convert numeric confidence to descriptive level"""
        if confidence > 0.9:
            return 'very_high'
        elif confidence > 0.7:
            return 'high'
        elif confidence > 0.5:
            return 'medium'
        else:
            return 'low'

    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive detection statistics

        Returns:
            Dictionary with detection performance statistics
        """
        total = max(self.prediction_stats['total_predictions'], 1)

        return {
            'total_predictions': self.prediction_stats['total_predictions'],
            'bot_rate': self.prediction_stats['bot_predictions'] / total,
            'human_rate': self.prediction_stats['human_predictions'] / total,
            'high_confidence_rate': self.prediction_stats['high_confidence_predictions'] / total,
            'low_agreement_rate': self.prediction_stats['low_agreement_predictions'] / total,
            'model_info': {
                'model_type': 'cls_maxpool_ensemble',
                'primary_pooling': self.config.ensemble.primary_pooling,
                'backup_pooling': self.config.ensemble.backup_pooling,
                'combination_method': self.config.ensemble.combination_method,
                'device': str(self.device)
            }
        }

    def reset_statistics(self):
        """Reset prediction statistics"""
        self.prediction_stats = {
            'total_predictions': 0,
            'bot_predictions': 0,
            'human_predictions': 0,
            'high_confidence_predictions': 0,
            'low_agreement_predictions': 0
        }


# Factory function for easy integration
def create_ensemble_bot_detector(model_path: Optional[str] = None,
                                config_override: Optional[Dict] = None) -> EnsembleBotDetector:
    """
    Factory function to create ensemble bot detector

    Args:
        model_path: Path to trained model (optional)
        config_override: Configuration overrides (optional)

    Returns:
        Configured EnsembleBotDetector instance
    """
    return EnsembleBotDetector(model_path=model_path, config_override=config_override)