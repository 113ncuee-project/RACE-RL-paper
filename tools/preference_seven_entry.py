#!/usr/bin/env python3
"""Run the existing CNN experiment entries for every supported frontend model.

Only the command-line model choices are extended; the original search,
workload, evaluator, hardware and calibration implementations are unchanged.
"""
import argparse
import sys

from race_rl.frontend import SUPPORTED_MODELS
import common_dse_trial
import explore_cnn_requirements


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--entry',choices=('trial','explore'),required=True)
    args,rest=parser.parse_known_args()
    original=argparse.ArgumentParser.add_argument
    def add_argument(self,*names,**kwargs):
        if '--model' in names:
            kwargs['choices']=SUPPORTED_MODELS
        return original(self,*names,**kwargs)
    argparse.ArgumentParser.add_argument=add_argument
    sys.argv=[sys.argv[0],*rest]
    (common_dse_trial.main if args.entry=='trial' else explore_cnn_requirements.main)()


if __name__=='__main__':main()
