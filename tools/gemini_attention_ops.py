"""Explicit, untrained Transformer tensor primitives in NCHW form."""
import math
import torch


def select_channels(x, start, width):
    return x[:, start:start+width, :, :].contiguous()


def attention_qk(q, k, heads):
    batch, channels, length, width = q.shape
    assert width == 1 and k.shape == q.shape and channels % heads == 0
    depth = channels // heads
    q = q.reshape(batch, heads, depth, length)
    k = k.reshape(batch, heads, depth, length)
    return torch.einsum('bhdk,bhdq->bhkq', k, q).reshape(batch, heads*length, length, 1)


def attention_softmax(scores, heads, causal):
    batch, channels, length, width = scores.shape
    assert width == 1 and channels == heads*length
    scores = scores.reshape(batch, heads, length, length)
    if causal:
        key = torch.arange(length, device=scores.device).reshape(length, 1)
        query = torch.arange(length, device=scores.device).reshape(1, length)
        scores = scores.masked_fill(key > query, -torch.inf)
    return torch.softmax(scores, dim=2).reshape(batch, channels, length, 1)


def attention_av(attn, value, heads):
    batch, channels, length, width = value.shape
    assert width == 1 and channels % heads == 0
    assert attn.shape == (batch, heads*length, length, 1)
    depth = channels // heads
    a = attn.reshape(batch, heads, length, length)
    v = value.reshape(batch, heads, depth, length)
    return torch.einsum('bhkq,bhdk->bhdq', a, v).reshape(batch, channels, length, 1)


def channel_norm(x, epsilon=1e-5):
    mean = x.mean(dim=1, keepdim=True)
    variance = ((x-mean)**2).mean(dim=1, keepdim=True)
    return (x-mean) * torch.rsqrt(variance + epsilon)
