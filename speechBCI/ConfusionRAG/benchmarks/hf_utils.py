"""Shared Hugging Face model-loading helpers for benchmark scripts."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Tuple


def add_hf_model_args(parser: argparse.ArgumentParser) -> None:
    """Add common HF runtime flags to a benchmark parser."""
    parser.add_argument(
        "--hf_device_map",
        type=str,
        default="auto",
        help=(
            "Hugging Face device map. Use 'auto' for automatic placement, "
            "or 'none' to disable device_map."
        ),
    )
    parser.add_argument(
        "--hf_torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--hf_trust_remote_code",
        action="store_true",
        help="Enable trust_remote_code for custom model/tokenizer repos.",
    )
    parser.add_argument(
        "--hf_attn_implementation",
        type=str,
        default=None,
        help="Optional attention implementation override (e.g. flash_attention_2).",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=0,
        help=(
            "Maximum number of hypotheses scored per forward pass "
            "(<=0 means score all at once)."
        ),
    )


def _resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def _build_model_kwargs(args) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(args.hf_trust_remote_code),
    }

    device_map = (args.hf_device_map or "").strip().lower()
    if device_map and device_map not in {"none", "null", "false"}:
        kwargs["device_map"] = args.hf_device_map

    kwargs["torch_dtype"] = _resolve_torch_dtype(args.hf_torch_dtype)

    if args.hf_attn_implementation:
        kwargs["attn_implementation"] = args.hf_attn_implementation

    return kwargs


def load_hf_causal_lm(llm_name: str, args) -> Tuple[Any, Any]:
    """Load a tokenizer + causal LM with robust defaults."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        llm_name,
        trust_remote_code=bool(args.hf_trust_remote_code),
    )
    tok.padding_side = "right"

    added_pad_token = False
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            added_pad_token = True

    model_kwargs = _build_model_kwargs(args)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            **model_kwargs,
        )
    except ValueError as exc:
        if "requires `accelerate`" in str(exc) and "device_map" in model_kwargs:
            # Graceful fallback for environments without accelerate.
            model_kwargs = dict(model_kwargs)
            model_kwargs.pop("device_map", None)
            model = AutoModelForCausalLM.from_pretrained(
                llm_name,
                **model_kwargs,
            )
        else:
            raise

    if added_pad_token:
        model.resize_token_embeddings(len(tok))

    return model, tok
