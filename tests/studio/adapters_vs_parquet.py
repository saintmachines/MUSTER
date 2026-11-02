#!/usr/bin/env python3
"""G0a+ (adapters): rebuild the LoCoMo store texts from the benchmark's own json with muster.adapters.locomo and
muster.serialize, and compare them with the contexts the published runs read (beam/locomo.parquet).

  EXPERIMENT_DIR=<tree> python tests/studio/adapters_vs_parquet.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT]
import pandas as pd

import muster
from muster.adapters import locomo

M = os.environ["EXPERIMENT_DIR"]
D = json.load(open(os.path.join(M, "locomo", "locomo10.json")))
df = pd.read_parquet(os.path.join(M, "beam", "locomo.parquet"))
same = 0
for i, d in enumerate(D):
    text, _ = muster.serialize(locomo(d["conversation"]), normalize=False)
    ok = text == str(df["context"].iloc[i])
    same += ok
    print("conversation %d: %s (%d characters)" % (i, "equal" if ok else "DIFFERENT", len(text)))
print("LoCoMo adapter: %d of %d store texts equal to the published parquet" % (same, len(D)))
sys.exit(0 if same == len(D) else 1)
