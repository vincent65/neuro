#!/usr/bin/env bash
# Run neural inference on GPU. Sets LD_LIBRARY_PATH so TensorFlow 2.7 finds CUDA 11 libs.

set -e
cd "$(dirname "$0")"
source speechBCI/.venv-decode/bin/activate

# TensorFlow 2.7 needs CUDA 11 libs from pip packages; add them to library path
VENV_LIBS="speechBCI/.venv-decode/lib/python3.9/site-packages/nvidia"
export LD_LIBRARY_PATH="${VENV_LIBS}/cuda_runtime/lib:${VENV_LIBS}/cublas/lib:${VENV_LIBS}/cudnn/lib:${VENV_LIBS}/cuda_nvrtc/lib:${VENV_LIBS}/cufft/lib:${VENV_LIBS}/curand/lib:${VENV_LIBS}/cusolver/lib:${VENV_LIBS}/cusparse/lib:${LD_LIBRARY_PATH}"

python run_inference_gpu.py "$@"
