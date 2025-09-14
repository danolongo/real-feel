"""
Configuration module for CLS + MaxPool Ensemble Bot Detection Transformer
Based on insights from experiments 3, 4, and 5
"""

import torch
from dataclasses import dataclass
# from typing import Literal

@dataclass
class ModelConfig:
    """model architecture config"""
    d_model: int = 512
    num_layers: int = 9
    num_heads: int = 8  # 512 / 8 = 64 (they have to be divisible)
    d_ff: int = 2048  # d_model * 4
    dropout: float = 0.15
    max_seq_length: int = 128
    vocab_size: int = 50265  # RoBERTa vocab size
    num_classes: int = 2  # binary, human or bot

@dataclass
class TrainingConfig:
    """Training configuration based on optimization findings"""
    optimizer_type: str = 'adamw'
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8

    # learning rate scheduling
    scheduler_type: str = 'cosine'
    warmup_steps: int = 1000

    # gradient clipping
    gradient_clipping: bool = True
    clip_type: str = 'norm'
    clip_value: float = 1.0

    # training parameters
    batch_size: int = 32
    max_epochs: int = 10

    # loss function (from experiment 4)
    use_class_weights: bool = True
    loss_type: str = 'weighted_ce'  # 'weighted_ce', 'focal', 'label_smoothed'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1

@dataclass
class DataConfig:
    """Data configuration"""
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    max_length: int = 128
    vocab_size: int = 50265
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class EnsembleConfig:
    """CLS + MaxPool ensemble configuration"""
    primary_pooling: str = 'cls'
    primary_weight: float = 0.7

    backup_pooling: str = 'max'
    backup_weight: float = 0.3

    # ensemble strategy
    combination_method: str = 'weighted_average'  # 'weighted_average', 'adaptive', 'voting'
    confidence_threshold: float = 0.8  # when to trust primary model alone

    # spam detection thresholds
    obvious_spam_threshold: float = 0.9  # MaxPool confidence for obvious cases
    subtle_bot_threshold: float = 0.6    # CLS confidence for sophisticated cases

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    ensemble: EnsembleConfig

    # System
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed: int = 42 # random number (Hitchhiker's Guide to the Galaxy reference aparently)
    experiment_name: str = "cls_maxpool_ensemble"
    debug: bool = False
    save_dir: str = "./experiments"

def get_default_config() -> ExperimentConfig:
    """Get default configuration for CLS + MaxPool ensemble"""
    return ExperimentConfig(
        model=ModelConfig(),
        training=TrainingConfig(),
        data=DataConfig(),
        ensemble=EnsembleConfig()
    )

def get_fast_config() -> ExperimentConfig:
    """Get configuration for fast experimentation"""
    config = get_default_config()
    config.model.num_layers = 6  # reduced for speed
    config.model.num_heads = 8
    config.training.max_epochs = 5
    config.training.batch_size = 16
    config.debug = True
    
    return config

def get_production_config() -> ExperimentConfig:
    """Get configuration optimized for production"""
    config = get_default_config()
    config.model.num_layers = 12  # more layers for better performance
    config.training.max_epochs = 20
    config.training.batch_size = 64
    config.training.gradient_clipping = True
    config.ensemble.primary_weight = 0.75  # trust CLS more in production

    return config