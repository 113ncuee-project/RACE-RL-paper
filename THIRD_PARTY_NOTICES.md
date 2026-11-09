# Third-party notices

Third-party files keep their original copyright notices and licenses. No third-party file is relicensed by this repository.

| Path | Source | License |
|---|---|---|
| `third_party/gemmini-reference/` | [ucb-bar/gemmini](https://github.com/ucb-bar/gemmini), commit `8c3f9923a44a2fe2c7930587be297d6d4f8c09ca`. Read-only reference files for the default configuration. | Original `LICENSE` included (UC Regents) |
| `third_party/facenet_pytorch/` | Inception-ResNet-v1 blocks from [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch), author Timothy Esler | MIT, notice retained |
| `third_party/cadene_pnasnet/` | PNASNet from [Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch), Remi Cadene and contributors | BSD 3-Clause, notice retained |

The Gemmini-derived values parameterize an analytical model. This project did not compile or run the RTL snapshot, and the results do not represent validation or endorsement by the Gemmini authors. The LB/RF organization, FP32 vector unit, and inter-chiplet model are research extensions.

The adapted Gemini baseline in `tools/gemini_common_dse.py` is a restricted adaptation to this project's shared action and evaluation interface. It is not the original Gemini implementation.

No pretrained model weights are distributed.
