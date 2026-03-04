import torch
import torch.nn as nn
from transformers import AutoModel


class DistilBertClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        dropout: float = 0.2,
        pooling: str = "cls",     # "cls" = take token 0, "mean" = mean pool over tokens
        freeze_encoder: bool = False,
    ):
        super().__init__()

        # Load pretrained transformer encoder (DistilBERT backbone)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # embedding size from the backbone

        # Classification head settings
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)  # final classifier layer

        # Optional: freeze the encoder for faster training / feature extraction
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Forward pass through transformer encoder
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state  # token embeddings: [B, T, H]

        if self.pooling == "mean":
            # Mean pool across non-padding tokens using the attention mask
            mask = attention_mask.unsqueeze(-1).type_as(x)  # [B, T, 1]
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            # "CLS-like" pooling: take the first token representation (index 0)
            x = x[:, 0, :]

        x = self.dropout(x)       # regularization before classification
        return self.fc(x)         # logits for each class