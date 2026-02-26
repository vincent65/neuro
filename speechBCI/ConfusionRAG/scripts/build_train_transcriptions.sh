#!/usr/bin/env bash
# Extract unique train transcriptions from competitionData .mat files.
# Produces: artifacts/train_transcriptions.txt

set -e
cd "$(dirname "$0")/../../.."  # neuro
source speechBCI/.venv-decode/bin/activate

python - <<'PY'
import glob
import scipy.io as sio
from pathlib import Path

root = Path("/home/vincentyip/neuro")
train_mats = sorted(glob.glob(str(root / "competitionData/train/*.mat")))
out_path = root / "speechBCI/ConfusionRAG/artifacts/train_transcriptions.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)

lines = []
for p in train_mats:
    d = sio.loadmat(p)
    sents = d["sentenceText"]
    for s in sents:
        t = str(s).strip()
        if t:
            lines.append(t)

# dedupe while preserving order
seen = set()
uniq = []
for t in lines:
    if t not in seen:
        seen.add(t)
        uniq.append(t)

with open(out_path, "w") as f:
    for t in uniq:
        f.write(t + "\n")

print("Saved:", out_path)
print("Num lines:", len(uniq))
PY
