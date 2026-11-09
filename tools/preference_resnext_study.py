#!/usr/bin/env python3
"""Run the calibrated preference study for a 32-group ResNeXt-50 variant."""
import hashlib
from pathlib import Path

import preference_additional_models as study
from gemini_resnext_entry import MODEL

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'tools/preference_additional_models.py'
EXPECTED_RUNNER_SHA256 = '56486f92f859bc1c0ba4a7623c41e81e6f4ec0562d9f0e2ec5ee977d74cea529'
if hashlib.sha256(RUNNER.read_bytes()).hexdigest() != EXPECTED_RUNNER_SHA256:
    raise RuntimeError('base preference runner changed; inspect before starting or resuming')

study.ADDITIONAL = (MODEL,)
study.ENTRY = ROOT / 'tools/gemini_resnext_entry.py'
study.PLOT = ROOT / 'tools/plot_preference_resnext.py'
# The delegated runner hashes __file__ for resume; freeze this wrapper as well.
study.__file__ = __file__

if __name__ == '__main__':
    study.main()
