# coding=utf-8
"""Online Qwen3-ASR + Qwen3-ForcedAligner example for Houmo vLLM services.
Output: word/character timestamps from Qwen3-ForcedAligner.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, TypedDict

VLLM_DEV_ROOT = Path(__file__).resolve().parents[2]
for path in (VLLM_DEV_ROOT / "Qwen3-ASR",):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

JSON_STDOUT = sys.stdout
sys.stdout = sys.stderr

import librosa
import numpy as np
import requests
import soundfile as sf
from transformers import AutoConfig

from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor

DEFAULT_AUDIO_PATH = "/home/useradmin/workspaces/test_data/audio/61-70968-0000.wav"
DEFAULT_ALIGNER_MODEL_DIR = "/home/useradmin/workspaces/models/v1.3.0/Qwen3-ForcedAligner-0.6B"
DEFAULT_ALIGNER_CHAT_TEMPLATE = """{% for message in messages %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% endif %}{% endfor %}{% endfor %}"""

class AlignmentSegment(TypedDict):
    text: str
    prompt: str
    audio: np.ndarray
    start_offset_s: float
    estimated_prompt_tokens: int
    start_index: int
    end_index: int

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR + ForcedAligner with per-char token timestamps")
    parser.add_argument("--audio", default=DEFAULT_AUDIO_PATH, help="Input audio path.")
    parser.add_argument("--asr-base-url", default="http://127.0.0.1:12324/v1")
    parser.add_argument("--aligner-base-url", default="http://127.0.0.1:12325")
    parser.add_argument("--api-key", default="sk-123", help="Bearer token")
    parser.add_argument("--asr-model-name", default="Qwen3-ASR-0.6B")
    parser.add_argument("--aligner-model-name", default="Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--aligner-model-dir", default=DEFAULT_ALIGNER_MODEL_DIR)
    parser.add_argument(
        "--aligner-chat-template",
        default=DEFAULT_ALIGNER_CHAT_TEMPLATE,
        help=(
            "Chat template for the ForcedAligner /tokenize and /pooling requests. "
            "It keeps the processor-generated text prompt, which already contains "
            "the audio placeholder and <timestamp> tokens. The server must be "
            "started with --trust-request-chat-template when this is set."
        ),
    )
    parser.add_argument("--asr-language", default="")
    parser.add_argument("--align-language", default="Chinese")
    parser.add_argument("--context", default="You are a helpful assistant.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--asr-chunk-seconds", type=float, default=15.0, help="Audio chunk length")
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--max-prefill", type=int, default=411)
    parser.add_argument("--token-safety-margin", type=int, default=16)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    return parser.parse_args()

def normalize_base_url(base_url: str, *, want_v1: bool) -> str:
    normalized = base_url.rstrip("/")
    if want_v1 and not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    if not want_v1 and normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized.rstrip("/")

def load_audio(audio_path: str, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)

def split_audio(audio: np.ndarray, sample_rate: int, chunk_seconds: float) -> list[dict[str, Any]]:
    if chunk_seconds <= 0:
        return [{"audio": audio, "start_s": 0.0, "duration_s": len(audio) / sample_rate}]
    chunk_samples = int(round(chunk_seconds * sample_rate))
    chunks = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        chunks.append({
            "audio": chunk,
            "start_s": start / sample_rate,
            "duration_s": len(chunk) / sample_rate
        })
    return chunks

def encode_audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()

def encode_audio_to_base64(audio: np.ndarray, sample_rate: int) -> str:
    return base64.b64encode(encode_audio_to_wav_bytes(audio, sample_rate)).decode("utf-8")

def post_json(session: requests.Session, url: str, api_key: str, payload: dict, timeout: int) -> dict:
    resp = session.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload, timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()

def transcribe_chunk(session: requests.Session, args: argparse.Namespace, base_url: str, audio: np.ndarray, idx: int) -> str:
    wav = encode_audio_to_wav_bytes(audio, args.sample_rate)
    data = {
        "model": args.asr_model_name,
        "response_format": "json",
        "temperature": "0",
        "max_completion_tokens": str(args.max_completion_tokens)
    }
    if args.context:
        data["prompt"] = args.context
    if args.asr_language.strip():
        data["language"] = args.asr_language.strip()

    resp = session.post(
        f"{base_url}/audio/transcriptions",
        headers={"Authorization": f"Bearer {args.api_key}"},
        files={"file": (f"chunk_{idx:04d}.wav", io.BytesIO(wav), "audio/wav")},
        data=data, timeout=args.request_timeout
    )
    resp.raise_for_status()
    return resp.json().get("text", "").strip()

def transcribe_audio_chunks(session: requests.Session, args: argparse.Namespace, base_url: str, chunks: list[dict]) -> list[dict]:
    records = []
    for i, c in enumerate(chunks):
        text = transcribe_chunk(session, args, base_url, c["audio"], i)
        records.append({
            "index": i, "audio": c["audio"], "start_s": c["start_s"],
            "duration_s": c["duration_s"], "text": text
        })
        print(f"[asr chunk {i}] start={c['start_s']:.3f}s duration={c['duration_s']:.3f}s text={text!r}", file=sys.stderr)
    return records

def split_alignment_units(aligner, text: str, lang: str) -> list[str]:
    word_list, _ = aligner.encode_timestamp(text, lang)
    return [str(unit) for unit in word_list if str(unit)]

def normalize_alignment_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r"[^\w\s]+", "", normalized, flags=re.UNICODE)
    return normalized.upper()

def join_alignment_units(units: list[str], lang: str) -> str:
    return " ".join(units)

def get_unit_weights(units):
    return [1]*len(units)

def get_audio_slice(audio, prefix_weights, start_idx, end_idx, sr):
    total = prefix_weights[-1]
    n = len(audio)
    s = round(prefix_weights[start_idx] / total * n)
    e = round(prefix_weights[end_idx] / total * n)
    if e <= s:
        e = s + 1
    return audio[s:e], s / sr

def estimate_prompt_tokens(processor, prompt, audio):
    inputs = processor(text=[prompt], audio=[audio], return_tensors="pt")
    return inputs.input_ids.shape[-1]

def load_processor(model_dir):
    try:
        return Qwen3ASRProcessor.from_pretrained(
            model_dir, 
            # fix_mistral_regex=True
        )
    except:
        return Qwen3ASRProcessor.from_pretrained(model_dir)

def build_segments(processor, aligner, audio, text, lang, sr, max_prefill, margin, global_start):
    units = split_alignment_units(aligner, text, lang)
    if not units:
        return [], [], []
    weights = get_unit_weights(units)
    prefix = [0]
    for w in weights:
        prefix.append(prefix[-1] + w)
    budget = max(1, max_prefill - margin)
    segs = []
    i = 0
    while i < len(units):
        best = None
        low = i+1
        high = len(units)
        while low <= high:
            mid = (low+high)//2
            t = join_alignment_units(units[i:mid], lang)
            a, ls = get_audio_slice(audio, prefix, i, mid, sr)
            _, p = aligner.encode_timestamp(t, lang)
            est = estimate_prompt_tokens(processor, p, a)
            if est <= budget:
                best = {
                    "text": t, "prompt": p, "audio": a,
                    "start_offset_s": global_start + ls,
                    "estimated_prompt_tokens": est,
                    "start_index": i, "end_index": mid
                }
                low = mid +1
            else:
                high = mid -1
        if not best:
            t = join_alignment_units(units[i:i+1], lang)
            a, ls = get_audio_slice(audio, prefix, i, i+1, sr)
            _, p = aligner.encode_timestamp(t, lang)
            est = estimate_prompt_tokens(processor, p, a)
            best = {
                "text": t, "prompt": p, "audio": a,
                "start_offset_s": global_start + ls,
                "estimated_prompt_tokens": est,
                "start_index": i, "end_index": i+1
            }
        segs.append(best)
        i = best["end_index"]
    return segs, units, prefix

def build_msgs(audio_b64, prompt):
    return [{
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            {"type": "text", "text": prompt}
        ]
    }]

def tokenize(session, args, base_url, b64, prompt):
    payload = {"model": args.aligner_model_name, "messages": build_msgs(b64, prompt), "add_generation_prompt": False}
    if args.aligner_chat_template:
        payload["chat_template"] = args.aligner_chat_template
    return post_json(session, f"{base_url}/tokenize", args.api_key, payload, args.request_timeout)["tokens"]

def pooling(session, args, base_url, b64, prompt):
    payload = {"model": args.aligner_model_name, "task": "token_classify", "messages": build_msgs(b64, prompt)}
    if args.aligner_chat_template:
        payload["chat_template"] = args.aligner_chat_template
    data = post_json(session, f"{base_url}/pooling", args.api_key, payload, args.request_timeout)
    return np.array(data["data"][0]["data"], np.float32)

def format_output_text(text: str) -> str:
    return re.sub(r"^\W+|\W+$", "", str(text)).upper()

def refine_segment(session, args, base_url, proc, aligner, full_audio, units, prefix, seg, lang, gstart):
    b64 = encode_audio_to_base64(seg["audio"], args.sample_rate)
    tids = tokenize(session, args, base_url, b64, seg["prompt"])
    if len(tids) <= args.max_prefill:
        return [(seg, tids, b64)]
    if seg["end_index"] - seg["start_index"] <=1:
        return [(seg, tids, b64)]
    m = (seg["start_index"] + seg["end_index"])//2
    out = []
    for s, e in [(seg["start_index"],m), (m, seg["end_index"])]:
        t = join_alignment_units(units[s:e], lang)
        a, ls = get_audio_slice(full_audio, prefix, s, e, args.sample_rate)
        _, p = aligner.encode_timestamp(t, lang)
        nseg = {
            "text": t, "prompt": p, "audio": a,
            "start_offset_s": gstart+ls,
            "estimated_prompt_tokens": estimate_prompt_tokens(proc,p,a),
            "start_index": s, "end_index": e
        }
        out.extend(refine_segment(session,args,base_url,proc,aligner,full_audio,units,prefix,nseg,lang,gstart))
    return out

def run_alignment(session, args, base_url, config, aligner, seg, tids, b64):
    try:
        logits = pooling(session, args, base_url, b64, seg["prompt"])
        if logits.ndim != 2:
            raise ValueError(f"Unexpected pooling output shape: {tuple(logits.shape)}")
        if logits.shape[0] != len(tids):
            raise ValueError(
                "Mismatch between /tokenize token count and /pooling output rows: "
                f"token_count={len(tids)}, pooling_rows={logits.shape[0]}."
            )

        preds = logits.argmax(axis=-1)
        timestamp_token_id = int(config.timestamp_token_id)
        timestamp_segment_time = float(config.timestamp_segment_time)
        timestamp_ms = np.asarray(
            [
                int(pred) * timestamp_segment_time
                for token_id, pred in zip(tids, preds)
                if int(token_id) == timestamp_token_id
            ],
            dtype=np.float32,
        )

        word_list, _ = aligner.encode_timestamp(seg["text"], args.align_language)
        expected_timestamps = len(word_list) * 2
        if len(timestamp_ms) != expected_timestamps:
            raise ValueError(
                f"Expected {expected_timestamps} timestamp predictions, got {len(timestamp_ms)}"
            )

        base_start = float(seg["start_offset_s"])
        return [
            {
                "text": format_output_text(str(item["text"])),
                "start_time": round(base_start + float(item["start_time"]) / 1000.0, 3),
                "end_time": round(base_start + float(item["end_time"]) / 1000.0, 3),
            }
            for item in aligner.parse_timestamp(word_list, timestamp_ms)
        ]
    except Exception as e:
        print(f"[WARN] align skip: {seg['text'][:20]}..., error={e}", file=sys.stderr)
        return []

def align_records(session, args, base_url, records):
    proc = load_processor(args.aligner_model_dir)
    aligner = Qwen3ForceAlignProcessor()
    config = AutoConfig.from_pretrained(args.aligner_model_dir, trust_remote_code=True)
    data = []
    for rec in records:
        text = normalize_alignment_text(rec["text"])
        if not text:
            continue
        segs, units, prefix = build_segments(
            proc, aligner, rec["audio"], text, args.align_language,
            args.sample_rate, args.max_prefill, args.token_safety_margin, rec["start_s"]
        )
        exact = []
        for s in segs:
            exact.extend(refine_segment(session,args,base_url,proc,aligner,rec["audio"],units,prefix,s,args.align_language,rec["start_s"]))
        print(f"[align chunk {rec['index']}] units={len(units)} segments={len(exact)}", file=sys.stderr)
        for seg, tids, b64 in exact:
            data.extend(run_alignment(session,args,base_url,config,aligner,seg,tids,b64))
    return data

def main():
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    
    asr_url = normalize_base_url(args.asr_base_url, want_v1=True)
    # ====================== 这里已修复 ======================
    align_url = normalize_base_url(args.aligner_base_url, want_v1=False)
    
    audio = load_audio(str(audio_path), args.sample_rate)
    chunks = split_audio(audio, args.sample_rate, args.asr_chunk_seconds)
    print(f"audio={audio_path} duration={len(audio)/args.sample_rate:.2f}s chunks={len(chunks)}", file=sys.stderr)

    with requests.Session() as s:
        records = transcribe_audio_chunks(s, args, asr_url, chunks)
        data = align_records(s, args, align_url, records)

    result = {
        "data": data
    }
    sys.stdout = JSON_STDOUT
    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()