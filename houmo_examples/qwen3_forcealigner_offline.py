import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, TypedDict


os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import librosa
import numpy as np

from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor
from vllm import LLM


class AlignmentSegment(TypedDict):
    text: str
    prompt: str
    audio: np.ndarray
    start_offset_s: float
    estimated_prompt_tokens: int
    start_index: int
    end_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline word-level timestamp validation for Qwen3-ForcedAligner."
    )
    parser.add_argument(
        "--model-dir",
        default="/opt/models/Qwen3-ForcedAligner-0.6B",
        help="Forced aligner model directory.",
    )
    parser.add_argument(
        "--audio",
        default="/opt/test_data/audio/test.mp3",
        help="Input audio path.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Reference transcript used for forced alignment.",
    )
    parser.add_argument(
        "--language",
        default="Chinese",
        help="Language name passed to Qwen3ForceAlignProcessor.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for audio loading.",
    )
    parser.add_argument(
        "--disable-mm-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable mm processor cache. Enabled by default for current Houmo workaround.",
    )
    parser.add_argument(
        "--auto-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically split long audio/text requests into smaller alignment segments.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def load_audio(audio_path: str, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def build_llm(args: argparse.Namespace) -> LLM:
    model_dir = Path(args.model_dir)
    llm_kwargs = {
        "model": str(model_dir),
        "runner": "pooling",
        "enforce_eager": True,
        "hf_overrides": {
            "architectures": ["Qwen3ASRForcedAlignerForTokenClassification"]
        },
        "max_model_len": 411,
        "additional_config": {
            "visual_model_path": str(
                model_dir / "mmproj-Qwen3-ForcedAligner-Q4_0.gguf"
            )
        },
        "load_format": "dummy",
    }
    if args.disable_mm_cache:
        llm_kwargs["mm_processor_cache_gb"] = 0
    return LLM(**llm_kwargs)


def get_model_max_prefill(llm: LLM) -> int:
    model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    max_prefill = getattr(model, "max_prefill", None)
    if not isinstance(max_prefill, int):
        raise RuntimeError("Failed to read qwen3-forcealigner max_prefill from model")
    return max_prefill


def estimate_prompt_tokens(
    processor: Qwen3ASRProcessor,
    prompt: str,
    audio: np.ndarray,
) -> int:
    inputs = processor(
        text=[prompt],
        audio=[audio],
        return_tensors="pt",
        padding=True,
    )
    return int(inputs["input_ids"].shape[-1])


def split_text_units(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [char for char in normalized if not char.isspace()]


def get_unit_weights(units: list[str]) -> list[int]:
    return [1 for _ in units]


def get_audio_slice(
    audio: np.ndarray,
    prefix_weights: list[int],
    start_index: int,
    end_index: int,
    sample_rate: int,
) -> tuple[np.ndarray, float]:
    total_weight = prefix_weights[-1]
    if total_weight <= 0:
        raise ValueError("Failed to split alignment request because text is empty")

    audio_len = len(audio)
    start_sample = round(prefix_weights[start_index] / total_weight * audio_len)
    end_sample = round(prefix_weights[end_index] / total_weight * audio_len)

    if end_index > start_index and end_sample <= start_sample:
        end_sample = min(audio_len, start_sample + 1)

    audio_slice = audio[start_sample:end_sample]
    if audio_slice.size == 0:
        raise ValueError(
            f"Failed to derive non-empty audio slice for text range [{start_index}, {end_index})"
        )

    start_offset_s = start_sample / sample_rate
    return audio_slice.astype(np.float32, copy=False), start_offset_s


def build_segments(
    processor: Qwen3ASRProcessor,
    aligner: Qwen3ForceAlignProcessor,
    audio: np.ndarray,
    text: str,
    language: str,
    sample_rate: int,
    max_prefill: int,
) -> list[AlignmentSegment]:
    units = split_text_units(text)
    if not units:
        raise ValueError("Reference transcript is empty after whitespace normalization")

    weights = get_unit_weights(units)
    prefix_weights = [0]
    for weight in weights:
        prefix_weights.append(prefix_weights[-1] + weight)

    segments: list[AlignmentSegment] = []
    start_index = 0

    while start_index < len(units):
        low = start_index + 1
        high = len(units)
        best: AlignmentSegment | None = None

        while low <= high:
            mid = (low + high) // 2
            segment_text = "".join(units[start_index:mid])
            segment_audio, start_offset_s = get_audio_slice(
                audio,
                prefix_weights,
                start_index,
                mid,
                sample_rate,
            )
            _, prompt = aligner.encode_timestamp(segment_text, language)
            estimated_tokens = estimate_prompt_tokens(processor, prompt, segment_audio)

            candidate: AlignmentSegment = {
                "text": segment_text,
                "prompt": prompt,
                "audio": segment_audio,
                "start_offset_s": start_offset_s,
                "estimated_prompt_tokens": estimated_tokens,
                "start_index": start_index,
                "end_index": mid,
            }

            if estimated_tokens <= max_prefill:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1

        if best is None:
            segment_text = "".join(units[start_index:start_index + 1])
            segment_audio, _ = get_audio_slice(
                audio,
                prefix_weights,
                start_index,
                start_index + 1,
                sample_rate,
            )
            _, prompt = aligner.encode_timestamp(segment_text, language)
            estimated_tokens = estimate_prompt_tokens(processor, prompt, segment_audio)
            raise ValueError(
                "Even the smallest alignment segment exceeds qwen3-forcealigner hardware prefill capacity: "
                f"segment_text={segment_text!r}, estimated_prompt_tokens={estimated_tokens}, max_prefill={max_prefill}."
            )

        segments.append(best)
        start_index = int(best["end_index"])

    return segments


def run_alignment_segment(
    llm: LLM,
    config: Any,
    aligner: Qwen3ForceAlignProcessor,
    segment: AlignmentSegment,
    language: str,
) -> dict[str, Any]:
    timestamp_token_id = config.timestamp_token_id
    timestamp_segment_time = config.timestamp_segment_time

    text = segment["text"]
    word_list, _ = aligner.encode_timestamp(text, language)
    output = llm.encode(
        [{"prompt": segment["prompt"], "multi_modal_data": {"audio": segment["audio"]}}],
        pooling_task="token_classify",
    )[0]

    logits = output.outputs.data
    token_ids = output.prompt_token_ids
    if logits.ndim != 2:
        raise ValueError(
            f"Unexpected pooling output shape: {tuple(logits.shape)}. "
            "Qwen3-ForcedAligner must return token-classification logits shaped "
            "[num_prompt_tokens, num_timestamp_bins]. A 1D output usually means "
            "HoumoPoolingModelRunner loaded the embedding fallback instead of "
            "Qwen3ASRForcedAlignerForTokenClassification."
        )
    if logits.shape[0] != len(token_ids):
        raise ValueError(
            "Mismatch between prompt token count and pooling output rows: "
            f"token_count={len(token_ids)}, pooling_rows={logits.shape[0]}."
        )
    predictions = logits.argmax(dim=-1)
    ts_predictions = [
        int(pred.item()) * timestamp_segment_time
        for tid, pred in zip(token_ids, predictions)
        if tid == timestamp_token_id
    ]
    expected_timestamps = len(word_list) * 2
    if len(ts_predictions) != expected_timestamps:
        raise RuntimeError(
            f"Expected {expected_timestamps} timestamp predictions, got {len(ts_predictions)}"
        )
    parsed = aligner.parse_timestamp(word_list, np.array(ts_predictions))

    start_offset_s = segment["start_offset_s"]
    return {
        "text": text,
        "start_offset_s": round(start_offset_s, 3),
        "estimated_prompt_tokens": int(segment["estimated_prompt_tokens"]),
        "num_prompt_tokens": len(token_ids),
        "word_timestamps": [
            {
                "text": item["text"],
                "start_s": round(start_offset_s + float(item["start_time"]) / 1000.0, 3),
                "end_s": round(start_offset_s + float(item["end_time"]) / 1000.0, 3),
                "duration_s": round(float(item["end_time"] - item["start_time"]) / 1000.0, 3),
            }
            for item in parsed
        ],
    }


def main() -> None:
    args = parse_args()
    audio = load_audio(args.audio, args.sample_rate)

    aligner = Qwen3ForceAlignProcessor()
    processor = Qwen3ASRProcessor.from_pretrained(
        args.model_dir,
        # fix_mistral_regex=True, # transformer 5.x 不支持该参数
    )

    llm = build_llm(args)
    # max_prefill = get_model_max_prefill(llm)
    max_prefill = 411
    _, prompt = aligner.encode_timestamp(args.text, args.language)
    estimated_prompt_tokens = estimate_prompt_tokens(processor, prompt, audio)
    if estimated_prompt_tokens > max_prefill and not args.auto_split:
        raise ValueError(
            "Input exceeds qwen3-forcealigner hardware prefill capacity: "
            f"estimated_prompt_tokens={estimated_prompt_tokens} > max_prefill={max_prefill}. "
            "Please split the long audio/transcript into shorter segments before alignment."
        )

    config = llm.llm_engine.vllm_config.model_config.hf_config
    if estimated_prompt_tokens > max_prefill:
        segments = build_segments(
            processor,
            aligner,
            audio,
            args.text,
            args.language,
            args.sample_rate,
            max_prefill,
        )
    else:
        segments = [{
            "text": args.text,
            "prompt": prompt,
            "audio": audio,
            "start_offset_s": 0.0,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "start_index": 0,
            "end_index": len(split_text_units(args.text)),
        }]

    segment_outputs = [
        run_alignment_segment(llm, config, aligner, segment, args.language)
        for segment in segments
    ]

    result = {
        "audio": str(Path(args.audio)),
        "audio_duration_s": round(float(len(audio) / args.sample_rate), 3),
        "reference_text": args.text,
        "language": args.language,
        "max_prefill": max_prefill,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "auto_split_used": len(segments) > 1,
        "num_segments": len(segments),
        "timestamp_segment_time_ms": float(config.timestamp_segment_time),
        "segments": segment_outputs,
        "word_timestamps": [
            word_timestamp
            for segment_output in segment_outputs
            for word_timestamp in segment_output["word_timestamps"]
        ],
    }

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()