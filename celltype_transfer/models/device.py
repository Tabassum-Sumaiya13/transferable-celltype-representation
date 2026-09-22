"""The one device rule for every training stage: a GPU when present, `--cpu` forces the fallback.

There is no GPU fork of the code - one code path, so a local number and a Kaggle number come from
the same function. Checkpoints are always SAVED on the CPU, so a GPU fit reopens on a CPU-only
machine. (This line used to be copied into five files.)
"""
import sys

import torch

DEV = torch.device('cuda' if (torch.cuda.is_available() and '--cpu' not in sys.argv) else 'cpu')
