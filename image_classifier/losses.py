"""Training objective, separate from architecture and optimization."""
from torch import nn


def build_loss(label_smoothing=.05):
    """Cross-entropy on logits, NOT on softmax probabilities.

    L = -sum_k q_k log softmax(logits)_k
    q_k = (1-epsilon) * 1[k=label] + epsilon/2.
    Smoothing discourages overconfident predictions; it does not calibrate them.
    Every image has one human/source-provided label. No pseudo-labels are used.
    """
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
