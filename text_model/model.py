import torch
import torch.nn as nn
from transformers import AutoModel


class DistilBertClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        dropout: float = 0.2,
        pooling: str = "cls",     # "cls" or "mean"
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state  # [B, T, H]

        if self.pooling == "mean":
            # mean pooling over non-pad tokens
            mask = attention_mask.unsqueeze(-1).type_as(x)  # [B, T, 1]
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            # "cls-like" token at position 0 (common for DistilBERT)
            x = x[:, 0, :]

        x = self.dropout(x)
        return self.fc(x)