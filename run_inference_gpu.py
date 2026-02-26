import argparse
import gc
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf

import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder


def _get_rss_gb() -> float:
    """Read current process RSS from /proc (Linux) in GB."""
    with open("/proc/self/status", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                kb = float(line.split()[1])
                return kb / (1024.0 * 1024.0)
    return 0.0


def _session_chunks(start: int, end: int, chunk_size: int) -> List[Tuple[int, int]]:
    chunks: List[Tuple[int, int]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + chunk_size - 1, end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + 1
    return chunks


def _manifest_path(out_dir: str, session_start: int, session_end: int) -> str:
    return os.path.join(out_dir, f"inference_manifest_s{session_start:02d}-{session_end:02d}.json")


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_run_args(
    base: str,
    ckpt_dir: str,
    out_dir: str,
    sess_start: int,
    sess_end: int,
    batch_size: Optional[int] = None,
    dataset_buffer_size: Optional[int] = None,
):
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "args.yaml"))
    cfg["loadDir"] = ckpt_dir
    cfg["outputDir"] = out_dir
    cfg["mode"] = "infer"
    cfg["loadCheckpointIdx"] = None
    cfg["testDir"] = "test"
    if batch_size is not None:
        cfg["batchSize"] = int(batch_size)

    dataset_cfg = cfg["dataset"]
    if dataset_buffer_size is not None:
        dataset_cfg["bufferSize"] = int(dataset_buffer_size)
    total_sessions = len(dataset_cfg["sessions"])
    if sess_start < 0 or sess_end >= total_sessions:
        raise ValueError(
            f"Chunk session range {sess_start}..{sess_end} is out of bounds for "
            f"available sessions 0..{total_sessions - 1}."
        )

    selected_indices = list(range(sess_start, sess_end + 1))

    def _slice_field(field_name: str):
        if field_name not in dataset_cfg or dataset_cfg[field_name] is None:
            return
        values = list(dataset_cfg[field_name])
        if len(values) == total_sessions:
            dataset_cfg[field_name] = [values[i] for i in selected_indices]

    # Critical: slice session-level lists so upstream only builds datasets for this chunk.
    for field in (
        "sessions",
        "dataDir",
        "datasetProbabilityVal",
        "datasetProbability",
        "datasetToLayerMap",
        "syntheticDataDir",
        "labelDir",
    ):
        _slice_field(field)

    # Force local tfRecords path for the selected sessions.
    dataset_cfg["dataDir"] = [f"{base}/derived/tfRecords"] * len(selected_indices)

    # In infer mode, only validation probability matters.
    dataset_cfg["datasetProbabilityVal"] = [1.0] * len(selected_indices)
    if "datasetProbability" in dataset_cfg and dataset_cfg["datasetProbability"] is not None:
        n = len(selected_indices)
        dataset_cfg["datasetProbability"] = [1.0 / n] * n

    # Keep mapping dense/contiguous to satisfy input layer construction logic.
    dataset_cfg["datasetToLayerMap"] = list(range(len(selected_indices)))

    return cfg


def _new_buffer() -> Dict[str, Any]:
    return {
        "logits": [],
        "logitLengths": [],
        "decodedSeqs": [],
        "editDistances": [],
        "seqErrorRate": [],
        "trueSeqLengths": [],
        "trueSeqs": [],
        "transcriptions": [],
        "nbest": [],
        "utterances": 0,
    }


def _clear_buffer(buf: Dict[str, Any]) -> None:
    for key in ("logits", "logitLengths", "decodedSeqs", "editDistances", "seqErrorRate", "trueSeqLengths", "trueSeqs", "transcriptions", "nbest"):
        buf[key].clear()
    buf["utterances"] = 0


def _finalize_inference_out(buf: Dict[str, Any], loss_type: str) -> Dict[str, Any]:
    if buf["utterances"] == 0:
        raise ValueError("Cannot finalize empty shard buffer.")

    inf_out: Dict[str, Any] = {
        "logits": [],
        "logitLengths": [],
        "decodedSeqs": [],
        "editDistances": [],
        "trueSeqLengths": [],
        "trueSeqs": [],
        "transcriptions": [],
        "seqErrorRate": [],
    }

    logits_flat = [l for batch in buf["logits"] for l in list(batch)]
    max_logit_length = max(l.shape[0] for l in logits_flat)
    padded_logits = [np.pad(l, [[0, max_logit_length - l.shape[0]], [0, 0]]) for l in logits_flat]
    inf_out["logits"] = np.stack(padded_logits, axis=0)
    inf_out["logitLengths"] = np.concatenate(buf["logitLengths"], axis=0)
    inf_out["decodedSeqs"] = np.concatenate(buf["decodedSeqs"], axis=0)
    inf_out["trueSeqLengths"] = np.concatenate(buf["trueSeqLengths"], axis=0)
    inf_out["trueSeqs"] = np.concatenate(buf["trueSeqs"], axis=0)
    inf_out["transcriptions"] = np.concatenate(buf["transcriptions"], axis=0)

    if loss_type == "ctc":
        inf_out["editDistances"] = np.concatenate(buf["editDistances"], axis=0)
        inf_out["cer"] = float(np.sum(inf_out["editDistances"]) / float(np.sum(inf_out["trueSeqLengths"])))
        inf_out.pop("seqErrorRate", None)
    elif loss_type == "ce":
        inf_out["seqErrorRate"] = np.concatenate(buf["seqErrorRate"], axis=0)
        inf_out["cer"] = inf_out["seqErrorRate"]
        inf_out.pop("editDistances", None)
    else:
        raise ValueError(f"Unsupported lossType '{loss_type}'")

    return inf_out


def _ensure_under_memory_ceiling(memory_ceiling_gb: float, stage: str) -> float:
    rss_gb = _get_rss_gb()
    print(f"[mem] {stage}: RSS={rss_gb:.2f} GB")
    if rss_gb > memory_ceiling_gb:
        raise RuntimeError(
            f"Memory ceiling exceeded at '{stage}': RSS={rss_gb:.2f} GB > {memory_ceiling_gb:.2f} GB. "
            "Reduce --sessions-per-chunk or --max-utterances-per-shard."
        )
    return rss_gb


def _worker_main(args: argparse.Namespace) -> None:
    base = args.base
    ckpt_dir = f"{base}/derived/rnns/baselineRelease"
    lm_dir = f"{base}/speech_5gram/lang_test"
    out_dir = f"{base}/speechBCI/ConfusionRAG/artifacts"
    os.makedirs(out_dir, exist_ok=True)

    summary: Dict[str, Any] = {
        "status": "running",
        "session_start": args.session_start,
        "session_end": args.session_end,
        "memory_ceiling_gb": args.memory_ceiling_gb,
        "max_utterances_per_shard": args.max_utterances_per_shard,
        "shards": [],
        "total_utterances": 0,
        "peak_rss_gb": 0.0,
        "loss_type": None,
    }

    def _write_summary(status: str, error: Optional[str] = None) -> None:
        summary["status"] = status
        if error is not None:
            summary["error"] = error
        _save_json(args.worker_summary_path, summary)

    try:
        run_args = _build_run_args(
            base,
            ckpt_dir,
            out_dir,
            args.session_start,
            args.session_end,
            batch_size=args.batch_size,
            dataset_buffer_size=args.dataset_buffer_size,
        )

        tf.compat.v1.reset_default_graph()
        nsd = NeuralSequenceDecoder(run_args)
        # Inference path does not use training iterators; dropping them prevents
        # background train-dataset prefetch from retaining host RAM.
        if hasattr(nsd, "trainDatasetIterators"):
            nsd.trainDatasetIterators = []
        if hasattr(nsd, "tfTrainDatasets"):
            nsd.tfTrainDatasets = []
        if hasattr(nsd, "trainDatasetSelector"):
            nsd.trainDatasetSelector = {}
        gc.collect()
        _ensure_under_memory_ceiling(args.memory_ceiling_gb, "after_nsd_init_cleanup")

        decoder = lmDecoderUtils.build_lm_decoder(
            lm_dir,
            acoustic_scale=0.5,
            nbest=args.nbest,
            beam=args.beam,
        )
        loss_type = str(run_args["lossType"])
        summary["loss_type"] = loss_type
        buffer = _new_buffer()
        shard_index = 0

        def flush_shard() -> None:
            nonlocal shard_index
            if buffer["utterances"] == 0:
                return

            rss_before_flush = _ensure_under_memory_ceiling(args.memory_ceiling_gb, "before_flush")
            inference_out = _finalize_inference_out(buffer, loss_type=loss_type)
            nbest_outputs = list(buffer["nbest"])
            if len(inference_out["transcriptions"]) != len(nbest_outputs):
                raise RuntimeError(
                    "Shard alignment mismatch: len(transcriptions) != len(nbest_outputs). "
                    f"{len(inference_out['transcriptions'])} vs {len(nbest_outputs)}"
                )

            inf_name = (
                f"inference_out_test_s{args.session_start:02d}-{args.session_end:02d}_"
                f"part{shard_index:03d}.npy"
            )
            nbest_name = (
                f"nbest_outputs_test_s{args.session_start:02d}-{args.session_end:02d}_"
                f"part{shard_index:03d}.npy"
            )
            inf_path = os.path.join(out_dir, inf_name)
            nbest_path = os.path.join(out_dir, nbest_name)
            np.save(inf_path, inference_out, allow_pickle=True)
            np.save(nbest_path, np.array(nbest_outputs, dtype=object), allow_pickle=True)

            rss_after_flush = _get_rss_gb()
            summary["peak_rss_gb"] = max(summary["peak_rss_gb"], rss_before_flush, rss_after_flush)
            summary["shards"].append(
                {
                    "shard_index": shard_index,
                    "session_start": args.session_start,
                    "session_end": args.session_end,
                    "utterances": int(buffer["utterances"]),
                    "inference_out": inf_name,
                    "nbest_outputs": nbest_name,
                    "rss_gb_before_flush": round(float(rss_before_flush), 3),
                    "rss_gb_after_flush": round(float(rss_after_flush), 3),
                }
            )
            print(
                f"Saved shard part{shard_index:03d}: "
                f"utterances={buffer['utterances']} RSS(after)={rss_after_flush:.2f} GB"
            )
            shard_index += 1

            _clear_buffer(buffer)
            gc.collect()
            _ensure_under_memory_ceiling(args.memory_ceiling_gb, "after_gc_flush")

        for dataset_idx, val_prob in enumerate(run_args["dataset"]["datasetProbabilityVal"]):
            if val_prob <= 0:
                continue

            layer_idx = run_args["dataset"]["datasetToLayerMap"][dataset_idx]
            for data in nsd.tfValDatasets[dataset_idx]:
                out = nsd._valStep(data, layer_idx)

                batch_logits = out["logits"].numpy()
                batch_logit_lengths = out["logitLengths"].numpy()
                batch_true_seqs = out["trueSeq"].numpy() - 1
                batch_true_seq_lengths = out["nSeqElements"].numpy()
                batch_transcriptions = out["transcription"].numpy()

                tmp = tf.sparse.to_dense(out["decodedStrings"][0], default_value=-1).numpy()
                padded_mat = np.zeros([tmp.shape[0], run_args["dataset"]["maxSeqElements"]], dtype=np.int32) - 1
                end = min(tmp.shape[1], run_args["dataset"]["maxSeqElements"])
                padded_mat[:, :end] = tmp[:, :end]

                buffer["logits"].append(batch_logits)
                buffer["logitLengths"].append(batch_logit_lengths)
                buffer["decodedSeqs"].append(padded_mat)
                buffer["trueSeqLengths"].append(batch_true_seq_lengths)
                buffer["trueSeqs"].append(batch_true_seqs)
                buffer["transcriptions"].append(batch_transcriptions)
                if loss_type == "ctc":
                    buffer["editDistances"].append(out["editDistance"].numpy())
                elif loss_type == "ce":
                    buffer["seqErrorRate"].append(np.asarray(out["seqErrorRate"].numpy()).reshape(-1))

                batch_inf = {
                    "logits": batch_logits,
                    "logitLengths": batch_logit_lengths,
                }
                batch_nbest = lmDecoderUtils.nbest_with_lm_decoder(
                    decoder,
                    batch_inf,
                    outputType="speech_sil",
                    rescore=not args.disable_rescore,
                    blankPenalty=np.log(7),
                )
                buffer["nbest"].extend(batch_nbest)

                batch_size = int(batch_logits.shape[0])
                buffer["utterances"] += batch_size
                summary["total_utterances"] += batch_size

                rss_batch = _ensure_under_memory_ceiling(args.memory_ceiling_gb, "post_batch_decode")
                summary["peak_rss_gb"] = max(summary["peak_rss_gb"], rss_batch)
                if buffer["utterances"] >= args.max_utterances_per_shard:
                    flush_shard()

        flush_shard()

        del decoder
        del nsd
        gc.collect()
        tf.keras.backend.clear_session()
        _write_summary("completed")
    except Exception as exc:
        _write_summary("failed", error=str(exc))
        raise


def _upsert_chunk(manifest: Dict[str, Any], chunk_record: Dict[str, Any]) -> None:
    target = (chunk_record["session_start"], chunk_record["session_end"])
    chunks = manifest.setdefault("chunks", [])
    for idx, existing in enumerate(chunks):
        if (existing.get("session_start"), existing.get("session_end")) == target:
            chunks[idx] = chunk_record
            return
    chunks.append(chunk_record)


def _chunk_completed_and_present(chunk: Dict[str, Any], out_dir: str) -> bool:
    if chunk.get("status") != "completed":
        return False
    shards = chunk.get("shards", [])
    if not shards:
        return False
    for shard in shards:
        inf_name = shard.get("inference_out")
        nbest_name = shard.get("nbest_outputs")
        if not inf_name or not nbest_name:
            return False
        if not os.path.exists(os.path.join(out_dir, inf_name)):
            return False
        if not os.path.exists(os.path.join(out_dir, nbest_name)):
            return False
    return True


def _orchestrator_main(args: argparse.Namespace) -> None:
    if args.sessions_per_chunk < 1:
        raise ValueError("--sessions-per-chunk must be >= 1")
    if args.session_start > args.session_end:
        raise ValueError("--session-start must be <= --session-end")
    if args.max_utterances_per_shard < 1:
        raise ValueError("--max-utterances-per-shard must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.dataset_buffer_size < 1:
        raise ValueError("--dataset-buffer-size must be >= 1")

    base = args.base
    out_dir = f"{base}/speechBCI/ConfusionRAG/artifacts"
    os.makedirs(out_dir, exist_ok=True)

    manifest_path = args.manifest_path or _manifest_path(out_dir, args.session_start, args.session_end)
    if args.resume and os.path.exists(manifest_path):
        manifest = _load_json(manifest_path)
    else:
        manifest = {
            "session_start": args.session_start,
            "session_end": args.session_end,
            "sessions_per_chunk": args.sessions_per_chunk,
            "memory_ceiling_gb": args.memory_ceiling_gb,
            "max_utterances_per_shard": args.max_utterances_per_shard,
            "batch_size": args.batch_size,
            "dataset_buffer_size": args.dataset_buffer_size,
            "nbest": args.nbest,
            "beam": args.beam,
            "disable_rescore": bool(args.disable_rescore),
            "chunks": [],
        }
    _save_json(manifest_path, manifest)

    chunks = _session_chunks(args.session_start, args.session_end, args.sessions_per_chunk)
    for chunk_start, chunk_end in chunks:
        existing_chunk = None
        for c in manifest.get("chunks", []):
            if c.get("session_start") == chunk_start and c.get("session_end") == chunk_end:
                existing_chunk = c
                break

        if args.resume and existing_chunk and _chunk_completed_and_present(existing_chunk, out_dir):
            print(f"Skipping completed chunk {chunk_start}..{chunk_end} (resume enabled).")
            continue

        worker_summary_path = os.path.join(
            out_dir, f".worker_summary_s{chunk_start:02d}-{chunk_end:02d}.json"
        )
        if os.path.exists(worker_summary_path):
            os.remove(worker_summary_path)

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--base",
            args.base,
            "--session-start",
            str(chunk_start),
            "--session-end",
            str(chunk_end),
            "--memory-ceiling-gb",
            str(args.memory_ceiling_gb),
            "--nbest",
            str(args.nbest),
            "--beam",
            str(args.beam),
            "--batch-size",
            str(args.batch_size),
            "--dataset-buffer-size",
            str(args.dataset_buffer_size),
            "--max-utterances-per-shard",
            str(args.max_utterances_per_shard),
            "--worker-summary-path",
            worker_summary_path,
        ]
        if args.disable_rescore:
            cmd.append("--disable-rescore")

        print(f"\n=== Running isolated worker for sessions {chunk_start}..{chunk_end} ===")
        proc = subprocess.run(cmd, check=False)
        if not os.path.exists(worker_summary_path):
            raise RuntimeError(
                f"Worker for chunk {chunk_start}..{chunk_end} did not produce summary file: "
                f"{worker_summary_path}"
            )
        worker_summary = _load_json(worker_summary_path)
        chunk_record = {
            "session_start": chunk_start,
            "session_end": chunk_end,
            "status": worker_summary.get("status", "failed"),
            "worker_exit_code": int(proc.returncode),
            "total_utterances": int(worker_summary.get("total_utterances", 0)),
            "peak_rss_gb": float(worker_summary.get("peak_rss_gb", 0.0)),
            "loss_type": worker_summary.get("loss_type"),
            "shards": worker_summary.get("shards", []),
        }
        if "error" in worker_summary:
            chunk_record["error"] = worker_summary["error"]

        _upsert_chunk(manifest, chunk_record)
        _save_json(manifest_path, manifest)

        if proc.returncode != 0 or chunk_record["status"] != "completed":
            raise RuntimeError(
                f"Chunk {chunk_start}..{chunk_end} failed with exit code {proc.returncode}. "
                f"Error: {chunk_record.get('error', 'unknown')}"
            )

        print(
            f"Chunk {chunk_start}..{chunk_end} completed: utterances={chunk_record['total_utterances']}, "
            f"peak_RSS={chunk_record['peak_rss_gb']:.2f} GB, shards={len(chunk_record['shards'])}"
        )

    print("\nSaved manifest:", manifest_path)


def main():
    parser = argparse.ArgumentParser(
        description="GPU inference with process isolation and memory-safe shard streaming."
    )
    parser.add_argument("--base", default="/home/vincentyip/neuro")
    parser.add_argument("--session-start", type=int, default=4)
    parser.add_argument("--session-end", type=int, default=18)
    parser.add_argument("--sessions-per-chunk", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size override for lower RAM use (upstream default is 64).",
    )
    parser.add_argument(
        "--dataset-buffer-size",
        type=int,
        default=1,
        help="Dataset shuffle buffer size override to reduce training-pipeline RAM in inference mode.",
    )
    parser.add_argument(
        "--max-utterances-per-shard",
        type=int,
        default=256,
        help="Flush shard after this many utterances to bound in-process RAM.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional manifest path. Defaults to artifacts/inference_manifest_sXX-YY.json.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from an existing manifest by skipping already completed chunks.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume and rebuild manifest from scratch.",
    )
    parser.add_argument(
        "--memory-ceiling-gb",
        type=float,
        default=110.0,
        help="Abort early if process RSS exceeds this value.",
    )
    parser.add_argument("--nbest", type=int, default=20, help="N-best size for LM decoder.")
    parser.add_argument("--beam", type=int, default=10, help="Beam size for LM decoder.")
    parser.add_argument(
        "--disable-rescore",
        action="store_true",
        help="Disable second-pass LM rescoring to reduce memory usage.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-summary-path", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.worker_summary_path:
            raise ValueError("--worker-summary-path is required in --worker mode.")
        _worker_main(args)
        return

    _orchestrator_main(args)


if __name__ == "__main__":
    main()
