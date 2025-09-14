"""
Transformer-based Bot Detection System for Twitter
BERT-style encoder-only architecture for Cresci-2017 dataset
"""

"""
Transformer-based Bot Detection System for Twitter
Implementation of BERT-style encoder-only architecture for Cresci-2017 dataset
"""

__version__ = "1.0.0"
__author__ = "Daniel Martinez"
__description__ = "Transformer-based Twitter bot detection using Cresci-2017 dataset"

from .config import (
    ModelConfig, 
    TrainingConfig, 
    DataConfig, 
    ExperimentConfig,
    get_default_config,
    get_small_config,
    get_large_config
)

from .model import (
    BotDetectionTransformer,
    MultiHeadAttention,
    TransformerEncoderLayer,
    create_model
)

from .data import (
    TwitterTokenizer,
    TwitterBotDataset,
    create_data_loaders,
    load_cresci_demo_data
)

from .trainer import (
    BotDetectionTrainer,
    create_trainer
)

__all__ = [
    # Config
    'ModelConfig',
    'TrainingConfig', 
    'DataConfig',
    'ExperimentConfig',
    'get_default_config',
    'get_small_config',
    'get_large_config',
    
    # Model
    'BotDetectionTransformer',
    'MultiHeadAttention',
    'TransformerEncoderLayer',
    'create_model',
    
    # Data
    'TwitterTokenizer',
    'TwitterBotDataset',
    'create_data_loaders',
    'load_cresci_demo_data',
    
    # Training
    'BotDetectionTrainer',
    'create_trainer'
]