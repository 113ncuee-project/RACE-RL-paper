#!/usr/bin/env python3
"""Load PNASNet-5-Large without downloading pretrained weights.

The classifier flatten uses torch.flatten instead of view and size; it is
numerically equivalent and makes the inference graph explicit to the importer.
"""
import argparse
import sys
import torch

from race_rl.accelerators import cnn_search
from cadene_pnasnet.pnasnet import PNASNet5Large
import common_dse_trial
import explore_cnn_requirements

MODEL = 'pnasnet5large_gemini'


class PNASNet(PNASNet5Large):
    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.last_linear(x)


def make_model(name, seed=7):
    if name != MODEL:
        raise ValueError(f'unsupported model {name!r}')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PNASNet(num_classes=1001)
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
