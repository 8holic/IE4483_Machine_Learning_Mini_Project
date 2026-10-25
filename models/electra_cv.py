"""ELECTRA-base shortcut for finetune_cv.py (BERT architecture, replaced-token-detection objective).

Usage:
    python models/electra_cv.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finetune_cv

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--model-name", "google/electra-base-discriminator", "--out", "results/electra_results.png", *sys.argv[1:]]
    finetune_cv.main()
