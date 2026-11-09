#!/usr/bin/env python3
"""Run the normal CNN entries with a 32-group, width-2 ResNeXt-50 variant.

Gemini's artifact uses [3,4,6,3] bottlenecks, 32 groups, internal widths
[64,128,256,512], and downsamples on the first bottleneck convolution.
This TorchVision-based variant also includes explicit BN/ReLU operations;
it is therefore not asserted to be the identical Gemini artifact graph.
"""
import argparse
import sys

import torch
from torchvision.models.resnet import Bottleneck, ResNet

from race_rl.accelerators import cnn_search
import common_dse_trial
import explore_cnn_requirements

MODEL = 'resnext50_32x2d_gemini'


def make_model(name, seed=7):
    if name != MODEL:
        raise ValueError(f'unsupported model {name!r}')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=2)
    for block in model.modules():
        if isinstance(block, Bottleneck) and block.stride > 1:
            block.conv1.stride = (block.stride, block.stride)
            block.conv2.stride = (1, 1)
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
