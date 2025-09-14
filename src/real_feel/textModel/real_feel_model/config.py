"""
Configuration module for Bot Detection Transformer
Contains all hyperparameters and settings for the model
"""

import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """model hyperparameter configurations"""
    d_model: int = 512
    num_layers: int = 9
    num_heads: int = 12
    d_ff: int = d_model * 4
    dropout: int = 0.15
    max_seq_length: int = 128 # tweet char maximum is 280?
    
    # classification tasks
    num_classes: int = 2  # Binary: Human (0) vs Bot (1)
    num_multiclass_classes: int = 6  # only 6 out of the 9 datasets were usable
    pooling_strategy: str = 'cls'
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert self.d_model > 0, "d_model must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.num_heads > 0, "num_heads must be positive"

@dataclass
class TrainingConfig:
    """Complete configuration for training pipeline"""

    batch_size: int = 32
    learning_rate: int = 2e-5
    max_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: int = 0.01
    gradient_clip_norm: int = 1.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # class imbalance handling
    use_class_weights: bool = True
    focal_loss_alpha: float = 0.25  # for focal loss alternative
    focal_loss_gamma: float = 2.0
    
    # data
    vocab_size: int = 50265  # RoBERTa vocab size
    test_size: float = 0.2
    val_size: float = 0.1
    
    def __post_init__(self):
        """Validate training configuration"""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.max_epochs > 0, "max_epochs must be positive"
        assert self.scheduler_type in ["cosine", "linear", "constant"], "Invalid scheduler type"



@dataclass
class ModelTrainingConfig:
    d_model: int = 512
    num_layers: int = 9
    num_heads: int = 12
    d_ff: int = d_model * 4
    dropout: int = 0.15
    max_seq_length: int = 128 # tweet char maximum is 280?
    
    # classification tasks
    num_classes: int = 2  # Binary: Human (0) vs Bot (1)
    # num_multiclass_classes: int = 6  # only 6 out of the 9 datasets were usable
    pooling_strategy: str = 'cls'


    batch_size: int = 32
    learning_rate: int = 2e-5
    max_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: int = 0.01
    gradient_clip_norm: int = 1.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # class imbalance handling
    use_class_weights: bool = True
    focal_loss_alpha: float = 0.25  # for focal loss alternative
    focal_loss_gamma: float = 2.0
    
    # data
    vocab_size: int = 50265  # RoBERTa vocab size
    test_size: float = 0.2
    val_size: float = 0.1

    def __post_init__(self):
        """Validate configuration after initialization"""
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert self.d_model > 0, "d_model must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.max_epochs > 0, "max_epochs must be positive"
