#!/usr/bin/env python3
"""Prepare an explicit two-input, 8-head Transformer for the extended IR."""
import argparse
import gc
import sys
from dataclasses import replace

import torch

from race_rl.accelerators.cnn_quantization import calibrate, execute_quantized
from race_rl.frontend import from_torch
from race_rl.serialization import save_json
from gemini_transformer_model import GeminiShapeTransformer
import common_dse_trial
import explore_cnn_requirements

MODEL='transformer_gemini_shape'


def prepare_model(name, length, seed, folder):
    if name!=MODEL or not 2<=length<=512:
        raise ValueError('invalid Transformer model or sequence length')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=GeminiShapeTransformer().eval()
    # One physical input allocation contains encoder and decoder sequences.
    example=torch.randn(1,1024,length,1,generator=torch.Generator().manual_seed(seed+1),dtype=torch.float32)
    imported=from_torch(model,example,name)
    bundle=calibrate(imported,example)
    del imported,model;gc.collect()
    _,verification=execute_quantized(bundle,example.numpy())
    workload=replace(bundle.workload,source_metadata=(*bundle.workload.source_metadata,
        ('transformer_sequence_length',str(length)),('transformer_inputs','encoder and decoder hidden states; packed along channels'),
        ('transformer_architecture','6 encoders; 6 decoders; 8 heads; hidden 512; FFN 2048; 1000 outputs'),
        ('transformer_difference','explicit masked softmax and fixed-affine channel norm; GEMINI accounting graph omits these')))
    save_json(folder/'quantization.json',bundle.manifest)
    save_json(folder/'numerical_validation.json',verification)
    save_json(folder/'workload.json',workload)
    del bundle,example;gc.collect()
    return workload,verification


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--entry',choices=('trial','explore'),required=True)
    args,rest=p.parse_known_args()
    common_dse_trial.prepare_model=prepare_model
    explore_cnn_requirements.prepare_model=prepare_model
    original=argparse.ArgumentParser.add_argument
    def add_argument(self,*names,**kwargs):
        if '--model' in names:kwargs['choices']=(MODEL,)
        return original(self,*names,**kwargs)
    argparse.ArgumentParser.add_argument=add_argument
    sys.argv=[sys.argv[0],*rest]
    (common_dse_trial.main if args.entry=='trial' else explore_cnn_requirements.main)()


if __name__=='__main__':main()
