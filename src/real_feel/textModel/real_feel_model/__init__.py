"""
CLS + MaxPool Ensemble Bot Detection System
Advanced transformer-based approach combining sophisticated detection with obvious spam filtering
Based on insights from experiments 3, 4, and 5
"""

__version__ = "2.0.0"
__author__ = "Daniel Martinez"
__description__ = "CLS + MaxPool ensemble transformer for Twitter bot detection"

# Configuration
from .config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    EnsembleConfig,
    ExperimentConfig,
    get_default_config,
    get_fast_config,
    get_production_config
)

# Core Models
from .model import (
    BotDetectionTransformer,
    CLSMaxPoolEnsemble,
    AdvancedPoolingHead,
    MultiHeadAttention,
    TransformerEncoderLayer,
    create_ensemble_model
)

# Loss Functions
from .loss import (
    AdvancedLossFunction,
    EnsembleLoss,
    create_loss_function
)

# Optimization
from .optimizer import (
    OptimizationManager,
    AdvancedLRScheduler,
    AdvancedGradientClipper,
    OptimizerFactory
)

# Training
from .trainer import (
    EnsembleTrainer,
    create_ensemble_trainer
)

__all__ = [
    # Config
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'EnsembleConfig',
    'ExperimentConfig',
    'get_default_config',
    'get_fast_config',
    'get_production_config',

    # Models
    'BotDetectionTransformer',
    'CLSMaxPoolEnsemble',
    'AdvancedPoolingHead',
    'MultiHeadAttention',
    'TransformerEncoderLayer',
    'create_ensemble_model',

    # Loss Functions
    'AdvancedLossFunction',
    'EnsembleLoss',
    'create_loss_function',

    # Optimization
    'OptimizationManager',
    'AdvancedLRScheduler',
    'AdvancedGradientClipper',
    'OptimizerFactory',

    # Training
    'EnsembleTrainer',
    'create_ensemble_trainer'
]