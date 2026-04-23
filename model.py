"""
Hyuk Jin Chung
4/22/2026

MLP for gesture classification
- 2 hidden layers and output layer
"""

import torch.nn as nn

class HandGestureNet(nn.Module):
    """
    Gesture classifier
    - classify 5 gestures and "no class" for unknown gestures
    """
    def __init__(self):
        super(HandGestureNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(63, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # prevents overfitting to the relatively small dataset
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 6) # classes 0 through 5
        )

    def forward(self, x):
        return self.mlp(x)