import torch.nn as nn


def get_linear_head(in_features, num_classes):
    return nn.Linear(in_features, num_classes)


class MLPHead(nn.Module):
    def __init__(
        self, in_features, num_classes, hidden_dim=512, dropout=0.3, use_bn=True
    ):
        super().__init__()
        layers = [
            nn.Linear(in_features, hidden_dim),
        ]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
