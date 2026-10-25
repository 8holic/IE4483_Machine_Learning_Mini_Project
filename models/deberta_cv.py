"""DeBERTa-v3-base shortcut for finetune_cv.py. Run from the project root.

Usage:
    python models/deberta_cv.py
    python models/deberta_cv.py --epochs 4 --batch-size 16
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finetune_cv

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--model-name", "microsoft/deberta-v3-base", "--out", "results/deberta_results.png", *sys.argv[1:]]
    finetune_cv.main()
