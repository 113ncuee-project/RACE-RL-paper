#!/usr/bin/env python3
"""Use an explicit Inception-ResNet-v1 variant with the isolated extended IR.

The blocks follow facenet-pytorch (MIT); the five/ten/five residual stages and
1792-to-1792-to-1000 head follow the layer counts in the Gemini artifact.
Explicit BN, ReLU, concatenation and residual scaling are retained, so the
result is not claimed to be the exact simplified Gemini graph.
"""
import argparse
import sys

import torch
from torch import nn

from race_rl.accelerators import cnn_search
from facenet_pytorch.blocks import BasicConv2d, Block35, Block17, Block8, Mixed_6a, Mixed_7a
import common_dse_trial
import explore_cnn_requirements

MODEL = 'inception_resnet_v1_gemini'


class InceptionResNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            BasicConv2d(3, 32, 3, 2), BasicConv2d(32, 32, 3, 1),
            BasicConv2d(32, 64, 3, 1, padding=1), nn.MaxPool2d(3, stride=2),
            BasicConv2d(64, 80, 1, 1), BasicConv2d(80, 192, 3, 1),
            BasicConv2d(192, 256, 3, 2))
        self.a = nn.Sequential(*(Block35(scale=.17) for _ in range(5)))
        self.reduce_a = Mixed_6a()
        self.b = nn.Sequential(*(Block17(scale=.1) for _ in range(10)))
        self.reduce_b = Mixed_7a()
        self.c = nn.Sequential(*(Block8(scale=.2) for _ in range(5)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1792, 1792)
        self.fc2 = nn.Linear(1792, 1000)

    def forward(self, x):
        x = self.stem(x)
        x = self.a(x)
        x = self.reduce_a(x)
        x = self.b(x)
        x = self.reduce_b(x)
        x = self.c(x)
        return self.fc2(self.fc1(self.flatten(self.pool(x))))


def make_model(name, seed=7):
    if name != MODEL:
        raise ValueError(f'unsupported model {name!r}')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = InceptionResNetV1()
    return model.eval()


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--entry', choices=('trial', 'explore'), required=True)
    args, rest = p.parse_known_args()
    cnn_search.make_torchvision_model = make_model
    original = argparse.ArgumentParser.add_argument
    def add_argument(self, *names, **kwargs):
        if '--model' in names:
            kwargs['choices'] = (MODEL,)
        return original(self, *names, **kwargs)
    argparse.ArgumentParser.add_argument = add_argument
    sys.argv = [sys.argv[0], *rest]
    (common_dse_trial.main if args.entry == 'trial' else explore_cnn_requirements.main)()


if __name__ == '__main__':
    main()
