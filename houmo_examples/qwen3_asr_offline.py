# coding=utf-8
"""Offline Qwen3-ASR example for Houmo vLLM.

This script starts a local in-process vLLM engine for Qwen3-ASR and
transcribes a local audio file. It is the ASR counterpart of
``qwen3_forcealigner_offline.py`` in this directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


VLLM_DEV_ROOT = Path(__file__).resolve().parents[2]
for path in (
    VLLM_DEV_ROOT / "vllm",
    VLLM_DEV_ROOT / "vllm-houmo",
    VLLM_DEV_ROOT / "Qwen3-ASR",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

JSON_STDOUT = sys.stdout
sys.stdout = sys.stderr

os.environ.setdefault("HDPL_PLATFORM", "ASIC")
os.environ.setdefault("TCIM_BACKEND", "Xh2HalBackend")
os.environ.setdefault("HM_PLATFORM", "2")
os.environ.setdefault("LLAMA_LOG_VERBOSITY", "99999999")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import SAMPLE_RATE, normalize_audio_input


DEFAULT_MODEL_DIR = "/home/useradmin/workspaces/models/v1.3.0/Qwen3-ASR-0.6B"
DEFAULT_AUDIO_PATH = "/home/useradmin/workspaces/test_data/audio/61-70968-0000.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline Qwen3-ASR transcription with local Houmo vLLM backend."
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Qwen3-ASR model directory.")
    parser.add_argument(
        "--mmproj-path",
        default=None,
        help="Houmo ASR mmproj gguf path. Defaults to <model-dir>/mmproj-qwen3-asr.gguf.",
    )
    parser.add_argument("--audio", default=DEFAULT_AUDIO_PATH, help="Input audio path.")
    parser.add_argument(
        "--language",
        default="",
        help="Force output language; empty string enables auto language detection.",
    )
    parser.add_argument(
        "--context",
        default="You are a helpful assistant.",
        help="Optional ASR context or hot words.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens for each ASR request.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=15.0,
        help="Split audio before inference. Houmo Qwen3-ASR works best with <=15s chunks; <=0 disables manual splitting.",
    )
    parser.add_argument(
        "--disable-mm-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable vLLM multimodal processor cache. Enabled by default for current Houmo workflow.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def split_audio(audio: np.ndarray, chunk_seconds: float) -> list[dict[str, Any]]:
    if chunk_seconds <= 0:
        return [{"index": 0, "audio": audio, "start_s": 0.0, "duration_s": len(audio) / SAMPLE_RATE}]

    chunk_samples = max(1, int(round(chunk_seconds * SAMPLE_RATE)))
    chunks: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(audio), chunk_samples)):
        chunk = audio[start : start + chunk_samples]
        chunks.append(
            {
                "index": index,
                "audio": chunk,
                "start_s": start / SAMPLE_RATE,
                "duration_s": len(chunk) / SAMPLE_RATE,
            }
        )
    return chunks


def build_asr_model(args: argparse.Namespace) -> Qwen3ASRModel:
    model_dir = Path(args.model_dir)
    mmproj_path = Path(args.mmproj_path) if args.mmproj_path else model_dir / "mmproj-qwen3-asr.gguf"

    llm_kwargs: dict[str, Any] = {
        "model": str(model_dir),
        "load_format": "dummy",
        "additional_config": {"visual_model_path": str(mmproj_path)},
        "max_inference_batch_size": 1,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.disable_mm_cache:
        llm_kwargs["mm_processor_cache_gb"] = 0

    return Qwen3ASRModel.LLM(**llm_kwargs)


def transcribe_chunks(
    asr: Qwen3ASRModel,
    chunks: list[dict[str, Any]],
    context: str,
    language: str | None,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for chunk in chunks:
        result = asr.transcribe(
            audio=(chunk["audio"], SAMPLE_RATE),
            context=context,
            language=language,
            return_time_stamps=False,
        )[0]
        item = {
            "index": int(chunk["index"]),
            "start_s": round(float(chunk["start_s"]), 3),
            "duration_s": round(float(chunk["duration_s"]), 3),
            "language": result.language,
            "text": result.text,
        }
        print(
            f"[chunk {item['index']}] start={item['start_s']:.3f}s "
            f"duration={item['duration_s']:.3f}s language={result.language!r} text={result.text!r}",
            file=sys.stderr,
        )
        outputs.append(item)
    return outputs


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    audio = normalize_audio_input(str(audio_path)).astype(np.float32, copy=False)
    chunks = split_audio(audio, args.chunk_seconds)
    language = args.language.strip() or None

    print(
        f"starting Qwen3-ASR offline: model={args.model_dir} audio={audio_path} "
        f"duration={len(audio) / SAMPLE_RATE:.3f}s chunks={len(chunks)} language={language!r}",
        file=sys.stderr,
    )
    asr = build_asr_model(args)

    chunk_outputs = transcribe_chunks(asr, chunks, args.context, language)
    merged_text = "".join(str(item["text"] or "") for item in chunk_outputs)
    merged_language = ",".join(
        dict.fromkeys(str(item["language"]) for item in chunk_outputs if item.get("language"))
    )

    result = {
        "model": str(Path(args.model_dir)),
        "audio": str(audio_path),
        "audio_duration_s": round(float(len(audio) / SAMPLE_RATE), 3),
        "sample_rate": SAMPLE_RATE,
        "language": merged_language,
        "text": merged_text,
        "chunk_seconds": args.chunk_seconds,
        "num_chunks": len(chunks),
        "chunks": chunk_outputs,
    }

    sys.stdout = JSON_STDOUT
    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()