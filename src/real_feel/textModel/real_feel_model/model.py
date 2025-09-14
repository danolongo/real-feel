"""
CLS + MaxPool Ensemble Bot Detection Transformer
Based on conclusions from experiments 3, 4, and 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, Union

from config import ModelConfig, EnsembleConfig


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_projection = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        # Project to Q, K, V
        qkv = self.qkv_projection(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)

        q, k, v = qkv.chunk(3, dim=-1)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attended_values = torch.matmul(attention_weights, v)
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous()
        attended_values = attended_values.reshape(batch_size, seq_len, d_model)

        output = self.output_projection(attended_values)
        return output


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-norm configuration"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm self-attention
        attn_output = self.self_attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        # Pre-norm feed-forward
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x


class AdvancedPoolingHead(nn.Module):
    """
    Advanced pooling head supporting both CLS and MaxPool strategies
    Based on experiment 4 findings
    """

    def __init__(self, d_model: int, num_classes: int, pooling_strategy: str = 'cls', dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy

        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize classification head weights"""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def pool_representations(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool token representations into single sequence representation

        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len)

        Returns:
            pooled: (batch_size, d_model)
        """
        if self.pooling_strategy == 'cls':
            # Use [CLS] token (first token) - best for sophisticated bot detection
            pooled = hidden_states[:, 0, :]

        elif self.pooling_strategy == 'max':
            # Max pooling - best for obvious spam detection
            if attention_mask is not None:
                # Set padded positions to large negative value
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                masked_hidden = hidden_states.clone()
                masked_hidden[mask_expanded == 0] = -1e9
                pooled = masked_hidden.max(dim=1)[0]
            else:
                pooled = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        return pooled

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of pooling head

        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Pool representations
        pooled = self.pool_representations(hidden_states, attention_mask)

        # Apply layer norm and dropout
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)

        # Classification
        logits = self.classifier(pooled)

        return logits


class BotDetectionTransformer(nn.Module):
    """
    Single transformer model with configurable pooling strategy
    Base component for ensemble approach
    """

    def __init__(self, config: ModelConfig, pooling_strategy: str = 'cls'):
        super().__init__()
        self.config = config
        self.pooling_strategy = pooling_strategy

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.d_model)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.dropout
            ) for _ in range(config.num_layers)
        ])

        # Pooling head
        self.pooling_head = AdvancedPoolingHead(
            config.d_model,
            config.num_classes,
            pooling_strategy,
            config.dropout
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following BERT-style initialization"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Create position ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings

        # Scale embeddings (from BERT)
        embeddings = embeddings * math.sqrt(self.config.d_model)

        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to 4D mask for multi-head attention
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=embeddings.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        # Pass through transformer layers
        hidden_states = embeddings
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, extended_attention_mask)

        # Classification
        logits = self.pooling_head(hidden_states, attention_mask)

        return logits


class CLSMaxPoolEnsemble(nn.Module):
    """
    CLS + MaxPool Ensemble Bot Detection System
    Combines sophisticated detection (CLS) with obvious spam detection (MaxPool)
    """

    def __init__(self, model_config: ModelConfig, ensemble_config: EnsembleConfig):
        super().__init__()
        self.model_config = model_config
        self.ensemble_config = ensemble_config

        # Primary model: CLS pooling for sophisticated bot detection
        self.primary_model = BotDetectionTransformer(model_config, pooling_strategy='cls')

        # Backup model: MaxPool for obvious spam detection
        self.backup_model = BotDetectionTransformer(model_config, pooling_strategy='max')

        # Ensemble combination weights (learnable)
        if ensemble_config.combination_method == 'adaptive':
            self.combination_weights = nn.Parameter(torch.tensor([
                ensemble_config.primary_weight,
                ensemble_config.backup_weight
            ]))

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                return_individual: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through ensemble

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            return_individual: Whether to return individual model predictions

        Returns:
            If return_individual=False: ensemble logits (batch_size, num_classes)
            If return_individual=True: dict with 'ensemble', 'primary', 'backup' logits
        """
        # Get predictions from both models
        primary_logits = self.primary_model(input_ids, attention_mask)
        backup_logits = self.backup_model(input_ids, attention_mask)

        # Combine predictions based on ensemble strategy
        if self.ensemble_config.combination_method == 'weighted_average':
            # Simple weighted average
            ensemble_logits = (
                self.ensemble_config.primary_weight * primary_logits +
                self.ensemble_config.backup_weight * backup_logits
            )

        elif self.ensemble_config.combination_method == 'adaptive':
            # Learnable combination weights
            weights = F.softmax(self.combination_weights, dim=0)
            ensemble_logits = weights[0] * primary_logits + weights[1] * backup_logits

        elif self.ensemble_config.combination_method == 'confidence_gated':
            # Use backup model when primary is uncertain
            primary_probs = F.softmax(primary_logits, dim=-1)
            primary_confidence = torch.max(primary_probs, dim=-1)[0]

            # Use backup when primary confidence is low
            use_backup = primary_confidence < self.ensemble_config.confidence_threshold
            use_backup = use_backup.unsqueeze(-1).float()

            ensemble_logits = (
                (1 - use_backup) * primary_logits +
                use_backup * backup_logits
            )

        else:
            raise ValueError(f"Unknown combination method: {self.ensemble_config.combination_method}")

        if return_individual:
            return {
                'ensemble': ensemble_logits,
                'primary': primary_logits,
                'backup': backup_logits
            }
        else:
            return ensemble_logits

    def predict_with_reasoning(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Make prediction with reasoning about which model contributed most

        Returns:
            Dictionary with predictions, probabilities, and reasoning
        """
        with torch.no_grad():
            # Get individual predictions
            outputs = self.forward(input_ids, attention_mask, return_individual=True)

            # Convert to probabilities
            ensemble_probs = F.softmax(outputs['ensemble'], dim=-1)
            primary_probs = F.softmax(outputs['primary'], dim=-1)
            backup_probs = F.softmax(outputs['backup'], dim=-1)

            # Determine which model is more confident
            primary_confidence = torch.max(primary_probs, dim=-1)[0]
            backup_confidence = torch.max(backup_probs, dim=-1)[0]

            # Classification decisions
            ensemble_predictions = torch.argmax(ensemble_probs, dim=-1)
            primary_predictions = torch.argmax(primary_probs, dim=-1)
            backup_predictions = torch.argmax(backup_probs, dim=-1)

            return {
                'predictions': ensemble_predictions,
                'probabilities': ensemble_probs,
                'primary_confidence': primary_confidence,
                'backup_confidence': backup_confidence,
                'primary_predictions': primary_predictions,
                'backup_predictions': backup_predictions,
                'agreement': (primary_predictions == backup_predictions).float()
            }


def create_ensemble_model(model_config: ModelConfig, ensemble_config: EnsembleConfig) -> CLSMaxPoolEnsemble:
    """
    Factory function to create CLS + MaxPool ensemble model

    Args:
        model_config: Model architecture configuration
        ensemble_config: Ensemble strategy configuration

    Returns:
        Initialized ensemble model
    """
    model = CLSMaxPoolEnsemble(model_config, ensemble_config)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"CLS + MaxPool Ensemble Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Primary model (CLS): Sophisticated bot detection")
    print(f"  Backup model (MaxPool): Obvious spam detection")
    print(f"  Ensemble method: {ensemble_config.combination_method}")

    return model