"""
Advanced Loss Functions for Bot Detection
Based on experiment 4 findings on loss function design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
from typing import Optional, Tuple

class AdvancedLossFunction(nn.Module):
    """Advanced loss function with multiple strategies for class imbalance"""

    def __init__(self, num_classes: int = 2, class_weights: Optional[torch.Tensor] = None,
                    loss_type: str = 'weighted_ce', focal_alpha: float = 0.25,
                    focal_gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

        # Register class weights as buffer
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def compute_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute class weights from labels using inverse frequency

        Args:
            labels: Tensor of shape (N,)

        Returns:
            class_weights: Tensor of shape (num_classes,)
        """
        class_counts = torch.bincount(labels, minlength=self.num_classes).float()
        total_samples = labels.size(0)

        # Inverse frequency weighting: w_i = N / (n_classes * n_i)
        class_weights = total_samples / (self.num_classes * class_counts)

        # Handle zero counts (shouldn't happen in practice)
        class_weights = torch.where(class_counts == 0, torch.zeros_like(class_weights), class_weights)

        return class_weights

    def weighted_cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Weighted cross-entropy loss

        Theory: L = -∑(w_i * y_i * log(p_i))
        where w_i are class weights, y_i are true labels, p_i are predicted probabilities
        """
        if self.class_weights is None:
            self.class_weights = self.compute_class_weights(labels)

        return F.cross_entropy(logits, labels, weight=self.class_weights)

    def focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss for addressing class imbalance

        Theory: FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
        where p_t is the model's estimated probability for the true class
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t

        # Alpha weighting
        if isinstance(self.focal_alpha, (list, tuple, torch.Tensor)):
            alpha_t = self.focal_alpha[labels]
        else:
            alpha_t = self.focal_alpha

        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def label_smoothed_cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy with label smoothing

        Theory: Replaces hard targets with soft targets:
        y_smooth = (1-ε) * y_true + ε/K
        where ε is smoothing parameter, K is number of classes
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed labels
        smooth_labels = torch.zeros_like(log_probs)
        smooth_labels.fill_(self.label_smoothing / self.num_classes)
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing + self.label_smoothing / self.num_classes)

        loss = -torch.sum(smooth_labels * log_probs, dim=-1)
        return loss.mean()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss based on specified loss type

        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)

        Returns:
            loss: scalar tensor
        """
        if self.loss_type == 'weighted_ce':
            return self.weighted_cross_entropy(logits, labels)
        elif self.loss_type == 'focal':
            return self.focal_loss(logits, labels)
        elif self.loss_type == 'label_smoothed':
            return self.label_smoothed_cross_entropy(logits, labels)
        elif self.loss_type == 'standard_ce':
            return F.cross_entropy(logits, labels)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class EnsembleLoss(nn.Module):
    """
    Specialized loss function for ensemble training
    Balances individual model performance with ensemble coherence
    """

    def __init__(self, loss_config: dict, alpha: float = 0.7):
        """
        Args:
            loss_config: Configuration for base loss function
            alpha: Weight for primary vs ensemble loss (0.0 = only ensemble, 1.0 = only individual)
        """
        super().__init__()
        self.alpha = alpha

        # Individual model losses
        self.primary_loss = AdvancedLossFunction(**loss_config)
        self.backup_loss = AdvancedLossFunction(**loss_config)

        # Ensemble loss
        self.ensemble_loss = AdvancedLossFunction(**loss_config)

    def forward(self, outputs: dict, labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute ensemble loss

        Args:
            outputs: Dict with 'ensemble', 'primary', 'backup' logits
            labels: Ground truth labels

        Returns:
            total_loss: Combined loss
            loss_components: Dict with individual loss components
        """
        # Individual model losses
        primary_loss = self.primary_loss(outputs['primary'], labels)
        backup_loss = self.backup_loss(outputs['backup'], labels)
        ensemble_loss = self.ensemble_loss(outputs['ensemble'], labels)

        # Combine losses
        individual_loss = (primary_loss + backup_loss) / 2
        total_loss = self.alpha * individual_loss + (1 - self.alpha) * ensemble_loss

        loss_components = {
            'total': total_loss,
            'ensemble': ensemble_loss,
            'primary': primary_loss,
            'backup': backup_loss,
            'individual': individual_loss
        }

        return total_loss, loss_components


def create_loss_function(loss_type: str = 'weighted_ce', num_classes: int = 2,
                        class_weights: Optional[torch.Tensor] = None, **kwargs) -> AdvancedLossFunction:
    """
    Factory function to create loss function based on experiment 4 findings

    Args:
        loss_type: Type of loss ('weighted_ce', 'focal', 'label_smoothed')
        num_classes: Number of output classes
        class_weights: Optional pre-computed class weights
        **kwargs: Additional loss-specific parameters

    Returns:
        Configured loss function
    """
    return AdvancedLossFunction(
        num_classes=num_classes,
        class_weights=class_weights,
        loss_type=loss_type,
        **kwargs
    )