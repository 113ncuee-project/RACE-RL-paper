"""Six-encoder/six-decoder 8-head Transformer with two 64-token inputs.

The hidden sizes follow the Gemini artifact. Softmax, causal masking and
fixed-affine channel layer normalization are explicit additions relative to
Gemini's simplified accounting graph; no pretrained weights are used.
"""
import torch
from torch import nn
from torch.fx import wrap
from gemini_attention_ops import select_channels, attention_qk, attention_softmax, attention_av, channel_norm

wrap('select_channels')
wrap('attention_qk')
wrap('attention_softmax')
wrap('attention_av')
wrap('channel_norm')


class Attention(nn.Module):
    def __init__(self, d_model=512, heads=8, causal=False):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.q = nn.Conv2d(d_model, d_model, 1)
        self.k = nn.Conv2d(d_model, d_model, 1)
        self.v = nn.Conv2d(d_model, d_model, 1)
        self.out = nn.Conv2d(d_model, d_model, 1)
        self.scale = (d_model // heads) ** -.5

    def forward(self, q, kv):
        q, k, v = self.q(q), self.k(kv), self.v(kv)
        weights = attention_softmax(attention_qk(q, k, self.heads) * self.scale,
                                    self.heads, self.causal)
        return self.out(attention_av(weights, v, self.heads))


class TransformerBlock(nn.Module):
    def __init__(self, decoder=False):
        super().__init__()
        self.decoder = decoder
        self.attn = Attention(causal=decoder)
        self.cross = Attention() if decoder else None
        self.fc1 = nn.Conv2d(512, 2048, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(2048, 512, 1)

    def forward(self, x, encoded):
        x = channel_norm(x + self.attn(x, x))
        if self.decoder:
            x = channel_norm(x + self.cross(x, encoded))
        return channel_norm(x + self.fc2(self.relu(self.fc1(x))))


class GeminiShapeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList(TransformerBlock() for _ in range(6))
        self.decoder = nn.ModuleList(TransformerBlock(decoder=True) for _ in range(6))
        self.output = nn.Conv2d(512, 1000, 1)

    def forward(self, x):
        encoded = select_channels(x, 0, 512)
        decoded = select_channels(x, 512, 512)
        for layer in self.encoder:
            encoded = layer(encoded, encoded)
        for layer in self.decoder:
            decoded = layer(decoded, encoded)
        return self.output(decoded)
