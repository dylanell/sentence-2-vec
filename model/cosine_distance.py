"""
Cosine distance module.
"""


import torch
from torch.nn import Module


class CosineDistance(Module):
    def __init__(self):
        super(CosineDistance, self).__init__()

    def forward(self, x1, x2):
        """Compute batch-wise cosine distance between two batches of row vectors.

        Args:
            x1: batch of row vectors.
            x2: batch of row vectors.
        """
        return 1.0 - torch.nn.functional.cosine_similarity(x1, x2, dim=1)