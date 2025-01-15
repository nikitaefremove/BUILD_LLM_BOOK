import torch
import torch.nn as nn

from model.transformer_block import TransformerBlock
from model.layer_norm import LayerNorm


class GPTModel(nn.Module):
    """
    Implementation of the GPT (Generative Pre-trained Transformer) model.

    Args:
        cfg (dict): Configuration dictionary containing model parameters.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Attributes:
        tok_emb (nn.Embedding): Token embedding layer.
        pos_emb (nn.Embedding): Positional encoding layer.
        drop_emb (nn.Dropout): Dropout layer for token embeddings.
        trf_blocks (nn.Sequential): Sequential container of transformer blocks.
        final_norm (LayerNorm): Layer normalization before the output head.
        out_head (nn.Linear): Output linear layer.

    Methods:
        forward(in_idx): Forward pass through the model.
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg=cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        Forward pass through the model.

        Args:
            in_idx (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, seq_len, vocab_size).
        """

        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_emb = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits