import torch.nn as nn

from model.multi_head_attention import MultiHeadAttention
from model.feedforward import FeedForward
from model.layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    """
    A single transformer block consisting of two sub-layers: a multi-head self-attention mechanism and a position-wise feed-forward network.
    Each sub-layer is followed by residual connection and layer normalization.

    Args:
        cfg (dict): Configuration dictionary containing the following keys:
            - "emb_dim" (int): Dimensionality of the input embeddings.
            - "context_length" (int): Length of the context for attention mechanism.
            - "n_heads" (int): Number of attention heads.
            - "drop_rate" (float): Dropout rate.
            - "qkv_bias" (bool): Whether to use bias in query, key, and value projections.

    Attributes:
        att (MultiHeadAttention): Multi-head self-attention mechanism.
        ff (FeedForward): Position-wise feed-forward network.
        norm1 (LayerNorm): Layer normalization after the first sub-layer.
        norm2 (LayerNorm): Layer normalization after the second sub-layer.
        drop_shortcut (nn.Dropout): Dropout applied to the residual connection.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Forward pass through the transformer block.
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg=cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        Forward pass through the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, emb_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.att(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
