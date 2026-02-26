#!/usr/bin/env python3
"""
Merge chunked inference outputs back to monolithic files.

This script consumes the manifest emitted by run_inference_gpu.py and writes:
  - inference_out_test.npy
  - nbest_outputs_test.npy
so downstream benchmark scripts can keep their original interface.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _latest_manifest(artifacts_dir: Path) -> Path:
    manifests = sorted(
        artifacts_dir.glob("inference_manifest_s*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        raise FileNotFoundError(
            f"No manifest found in {artifacts_dir}. "
            "Run run_inference_gpu.py first or pass --manifest."
        )
    return manifests[0]


def _merge_value(left: Any, right: Any, key: str) -> Any:
    """Merge two chunk values for the same inference_out key."""
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.ndim == 0 and right.ndim == 0:
            if left.item() != right.item():
                raise ValueError(f"Scalar mismatch for key '{key}': {left.item()} vs {right.item()}")
            return left
        return np.concatenate([left, right], axis=0)

    if isinstance(left, list) and isinstance(right, list):
        return left + right

    if isinstance(left, tuple) and isinstance(right, tuple):
        return left + right

    if left == right:
        return left

    raise TypeError(
        f"Unsupported or incompatible value types for key '{key}': "
        f"{type(left)} vs {type(right)}"
    )


def merge_inference_dicts(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not dicts:
        raise ValueError("No inference shard dicts provided.")

    merged = dict(dicts[0])
    expected_keys = set(merged.keys())
    for idx, d in enumerate(dicts[1:], start=1):
        if set(d.keys()) != expected_keys:
            raise ValueError(f"Inference shard #{idx} keys do not match first shard keys.")
        for key in expected_keys:
            merged[key] = _merge_value(merged[key], d[key], key)
    return merged


def _extract_shards(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    shards: List[Dict[str, Any]] = []
    for chunk in chunks:
        if "shards" in chunk and chunk["shards"]:
            shards.extend(chunk["shards"])
        elif "inference_out" in chunk and "nbest_outputs" in chunk:
            # Backward compatibility for legacy manifest format.
            shards.append(
                {
                    "inference_out": chunk["inference_out"],
                    "nbest_outputs": chunk["nbest_outputs"],
                    "session_start": chunk.get("session_start"),
                    "session_end": chunk.get("session_end"),
                    "shard_index": 0,
                }
            )
    return shards


def _concatenate_scalar_or_vector(arrays: List[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.array([])
    if len(arrays[0].shape) == 0:
        return np.array([a.item() for a in arrays])
    return np.concatenate(arrays, axis=0)


def _two_pass_merge_inference(artifacts_dir: Path, shards: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not shards:
        raise ValueError("No shard entries available to merge.")

    total_utterances = 0
    global_max_logit_len = 0
    logits_classes = None
    decoded_width = None
    true_seq_width = None
    transcription_width = None
    logits_dtype = None
    logit_lengths_dtype = None
    decoded_dtype = None
    true_seq_lengths_dtype = None
    true_seqs_dtype = None
    transcriptions_dtype = None
    have_edit_distances = False
    have_seq_error_rate = False
    edit_dist_dtype = None
    seq_error_dtype = None

    for shard in shards:
        inf_path = artifacts_dir / shard["inference_out"]
        inference_out = np.load(inf_path, allow_pickle=True).item()
        logits = inference_out["logits"]
        n = int(logits.shape[0])
        total_utterances += n
        global_max_logit_len = max(global_max_logit_len, int(logits.shape[1]))

        if logits_classes is None:
            logits_classes = int(logits.shape[2])
            decoded_width = int(inference_out["decodedSeqs"].shape[1])
            true_seq_width = int(inference_out["trueSeqs"].shape[1])
            transcription_width = int(inference_out["transcriptions"].shape[1])
            logits_dtype = logits.dtype
            logit_lengths_dtype = inference_out["logitLengths"].dtype
            decoded_dtype = inference_out["decodedSeqs"].dtype
            true_seq_lengths_dtype = inference_out["trueSeqLengths"].dtype
            true_seqs_dtype = inference_out["trueSeqs"].dtype
            transcriptions_dtype = inference_out["transcriptions"].dtype
        else:
            if int(logits.shape[2]) != logits_classes:
                raise ValueError("Logit class dimension mismatch across shards.")
            if int(inference_out["decodedSeqs"].shape[1]) != decoded_width:
                raise ValueError("decodedSeqs width mismatch across shards.")
            if int(inference_out["trueSeqs"].shape[1]) != true_seq_width:
                raise ValueError("trueSeqs width mismatch across shards.")
            if int(inference_out["transcriptions"].shape[1]) != transcription_width:
                raise ValueError("transcriptions width mismatch across shards.")

        if "editDistances" in inference_out:
            have_edit_distances = True
            edit_dist_dtype = inference_out["editDistances"].dtype
        if "seqErrorRate" in inference_out:
            have_seq_error_rate = True
            seq_error_dtype = inference_out["seqErrorRate"].dtype

    merged: Dict[str, Any] = {
        "logits": np.zeros(
            (total_utterances, global_max_logit_len, logits_classes),
            dtype=logits_dtype,
        ),
        "logitLengths": np.zeros((total_utterances,), dtype=logit_lengths_dtype),
        "decodedSeqs": np.zeros((total_utterances, decoded_width), dtype=decoded_dtype),
        "trueSeqLengths": np.zeros((total_utterances,), dtype=true_seq_lengths_dtype),
        "trueSeqs": np.zeros((total_utterances, true_seq_width), dtype=true_seqs_dtype),
        "transcriptions": np.zeros((total_utterances, transcription_width), dtype=transcriptions_dtype),
    }
    if have_edit_distances:
        merged["editDistances"] = np.zeros((total_utterances,), dtype=edit_dist_dtype)
    if have_seq_error_rate:
        merged["seqErrorRate"] = np.zeros((total_utterances,), dtype=seq_error_dtype)

    write_idx = 0
    seq_error_parts: List[np.ndarray] = []
    for shard in shards:
        inf_path = artifacts_dir / shard["inference_out"]
        inference_out = np.load(inf_path, allow_pickle=True).item()
        logits = inference_out["logits"]
        n = int(logits.shape[0])
        if n == 0:
            continue

        end = write_idx + n
        merged["logits"][write_idx:end, : logits.shape[1], :] = logits
        merged["logitLengths"][write_idx:end] = inference_out["logitLengths"]
        merged["decodedSeqs"][write_idx:end, :] = inference_out["decodedSeqs"]
        merged["trueSeqLengths"][write_idx:end] = inference_out["trueSeqLengths"]
        merged["trueSeqs"][write_idx:end, :] = inference_out["trueSeqs"]
        merged["transcriptions"][write_idx:end, :] = inference_out["transcriptions"]

        if have_edit_distances and "editDistances" in inference_out:
            merged["editDistances"][write_idx:end] = inference_out["editDistances"]
        if have_seq_error_rate and "seqErrorRate" in inference_out:
            cur = np.asarray(inference_out["seqErrorRate"])
            if cur.ndim <= 1:
                merged["seqErrorRate"][write_idx:end] = cur.reshape(-1)[:n]
            else:
                seq_error_parts.append(cur)

        write_idx = end

    if write_idx != total_utterances:
        raise RuntimeError(
            f"Merged utterance count mismatch: wrote {write_idx}, expected {total_utterances}"
        )

    if have_edit_distances:
        denom = float(np.sum(merged["trueSeqLengths"]))
        merged["cer"] = float(np.sum(merged["editDistances"]) / denom) if denom > 0 else 0.0
    elif have_seq_error_rate:
        if seq_error_parts:
            merged["seqErrorRate"] = _concatenate_scalar_or_vector(seq_error_parts)
        merged["cer"] = merged["seqErrorRate"]

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge chunked inference shard outputs.")
    parser.add_argument(
        "--artifacts-dir",
        default="/home/vincentyip/neuro/speechBCI/ConfusionRAG/artifacts",
        help="Directory containing shard .npy files and manifest.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional explicit manifest path. If omitted, latest manifest in artifacts-dir is used.",
    )
    parser.add_argument("--out-inf-name", default="inference_out_test.npy")
    parser.add_argument("--out-nbest-name", default="nbest_outputs_test.npy")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else _latest_manifest(artifacts_dir)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    chunks = manifest.get("chunks", [])
    if not chunks:
        raise ValueError(f"Manifest has no chunks: {manifest_path}")

    shard_records = _extract_shards(chunks)
    if not shard_records:
        raise ValueError(f"Manifest has no shard records: {manifest_path}")

    # Stable merge order: session range then shard index.
    shard_records = sorted(
        shard_records,
        key=lambda s: (
            int(s.get("session_start", -1)),
            int(s.get("session_end", -1)),
            int(s.get("shard_index", 0)),
        ),
    )

    nbest_total = 0
    for i, shard in enumerate(shard_records):
        nbest_name = shard["nbest_outputs"]
        nbest_path = artifacts_dir / nbest_name
        inf_path = artifacts_dir / shard["inference_out"]
        if not inf_path.exists():
            raise FileNotFoundError(f"Missing inference shard file: {inf_path}")
        if not nbest_path.exists():
            raise FileNotFoundError(f"Missing nbest shard file: {nbest_path}")
        inference_out = np.load(inf_path, allow_pickle=True).item()
        nbest_outputs = np.load(nbest_path, allow_pickle=True)
        n_inf = len(inference_out.get("transcriptions", []))
        n_nbest = len(nbest_outputs)
        if n_inf != n_nbest:
            raise ValueError(
                f"Shard #{i} has length mismatch: len(transcriptions)={n_inf} vs len(nbest_outputs)={n_nbest}"
            )
        nbest_total += n_nbest

    nbest_merged = np.empty((nbest_total,), dtype=object)
    write_idx = 0
    for shard in shard_records:
        nbest_outputs = np.load(artifacts_dir / shard["nbest_outputs"], allow_pickle=True)
        end = write_idx + len(nbest_outputs)
        nbest_merged[write_idx:end] = nbest_outputs
        write_idx = end
    if write_idx != nbest_total:
        raise RuntimeError(f"N-best merge count mismatch: wrote {write_idx}, expected {nbest_total}")

    inference_merged = _two_pass_merge_inference(artifacts_dir, shard_records)

    out_inf = artifacts_dir / args.out_inf_name
    out_nbest = artifacts_dir / args.out_nbest_name
    np.save(out_inf, inference_merged, allow_pickle=True)
    np.save(out_nbest, nbest_merged, allow_pickle=True)

    print(f"Merged {len(shard_records)} shards from {manifest_path}")
    print(f"Saved: {out_inf}")
    print(f"Saved: {out_nbest}")
    print(f"Total utterances: {len(nbest_merged)}")


if __name__ == "__main__":
    main()
