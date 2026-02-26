#!/usr/bin/env bash
# Run neural decoder inference + 5-gram N-best on test split (sessions 4..18).
# Produces: artifacts/inference_out_test.npy, artifacts/nbest_outputs_test.npy

set -e
cd "$(dirname "$0")/../.."  # neuro/speechBCI
source .venv-decode/bin/activate

python - <<'PY'
import os
import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils

base = "/home/vincentyip/neuro"
ckpt_dir = f"{base}/derived/rnns/baselineRelease"
lm_dir = f"{base}/speech_5gram/lang_test"
out_dir = f"{base}/speechBCI/ConfusionRAG/artifacts"
os.makedirs(out_dir, exist_ok=True)

args = OmegaConf.load(os.path.join(ckpt_dir, "args.yaml"))
args["loadDir"] = ckpt_dir
args["outputDir"] = out_dir  # override Stanford path from args.yaml
args["mode"] = "infer"
args["loadCheckpointIdx"] = None
args["testDir"] = "test"

# Match the notebook behavior (sessions 4..18 on test split)
for i in range(len(args["dataset"]["datasetProbabilityVal"])):
    args["dataset"]["datasetProbabilityVal"][i] = 0.0
for sess_idx in range(4, 19):
    args["dataset"]["datasetProbabilityVal"][sess_idx] = 1.0
    args["dataset"]["dataDir"][sess_idx] = f"{base}/derived/tfRecords"

with tf.device("/CPU:0"):
    tf.compat.v1.reset_default_graph()
    nsd = NeuralSequenceDecoder(args)
    inference_out = nsd.inference()

# 5-gram decoder settings from the provided notebook
decoder = lmDecoderUtils.build_lm_decoder(
    lm_dir,
    acoustic_scale=0.5,
    nbest=100,
    beam=18
)
nbest_outputs = lmDecoderUtils.nbest_with_lm_decoder(
    decoder,
    inference_out,
    outputType="speech_sil",
    rescore=True,
    blankPenalty=np.log(7)
)

np.save(f"{out_dir}/inference_out_test.npy", inference_out, allow_pickle=True)
np.save(f"{out_dir}/nbest_outputs_test.npy", np.array(nbest_outputs, dtype=object), allow_pickle=True)
print("Saved:", f"{out_dir}/inference_out_test.npy")
print("Saved:", f"{out_dir}/nbest_outputs_test.npy")
PY
