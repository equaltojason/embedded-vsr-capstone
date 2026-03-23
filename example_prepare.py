import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

OUT_W = OUT_H = 96
PAD_FRAC = 0.35

LIPS_IDX = sorted(set([
    0, 13, 14, 37, 39, 40, 267, 269, 270, 61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 78, 95, 88, 178, 87, 317, 402, 318, 324
]))

MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/face_landmarker.task"

DOWNLOADS = Path.home() / "Downloads"
OUT_ROOT = Path.home() / "VSRProjLocal" / "example"

RAW_DIR = OUT_ROOT / "raw"
MOUTH_DIR = OUT_ROOT / "mouth"
AUDIO_DIR = OUT_ROOT / "audio"
TXT_DIR = OUT_ROOT / "transcripts"

def ensure_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        print("ffmpeg not found. Install with: brew install ffmpeg")
        sys.exit(1)

def cache_dir() -> Path:
    d = Path.home() / ".cache" / "vsr_pipeline"
    d.mkdir(parents=True, exist_ok=True)
    return d

def ensure_landmarker_model() -> Path:
    model_path = cache_dir() / "face_landmarker.task"
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    print(f"Downloading FaceLandmarker model to {model_path}")
    subprocess.run(["curl", "-L", "-o", str(model_path), MODEL_URL], check=True)
    return model_path

def make_landmarker(model_path: Path):
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def bbox_from_lips(landmarks, w, h, idxs, pad_frac):
    xs, ys = [], []
    for i in idxs:
        lm = landmarks[i]
        xs.append(lm.x * w)
        ys.append(lm.y * h)

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    bw, bh = (x1 - x0), (y1 - y0)
    pad_x, pad_y = bw * pad_frac, bh * pad_frac

    x0 = int(np.floor(x0 - pad_x))
    x1 = int(np.ceil(x1 + pad_x))
    y0 = int(np.floor(y0 - pad_y))
    y1 = int(np.ceil(y1 + pad_y))

    x0 = clamp(x0, 0, w - 1)
    x1 = clamp(x1, 1, w)
    y0 = clamp(y0, 0, h - 1)
    y1 = clamp(y1, 1, h)

    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return None
    return (x0, y0, x1, y1)

def fallback_bbox(w, h):
    cx, cy = w // 2, int(h * 0.62)
    size = int(min(w, h) * 0.35)
    x0 = clamp(cx - size // 2, 0, w - 2)
    y0 = clamp(cy - size // 2, 0, h - 2)
    x1 = clamp(x0 + size, 2, w)
    y1 = clamp(y0 + size, 2, h)
    return (x0, y0, x1, y1)

def extract_audio_ffmpeg(video_path: Path, wav_path: Path):
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        str(wav_path)
    ]
    subprocess.run(cmd, check=True)

def crop_mouth_video(video_path: Path, out_path: Path, model_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 25.0
    frame_period_ms = 1000.0 / fps

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (OUT_W, OUT_H)
    )

    landmarker = make_landmarker(model_path)
    prev_bbox = None
    frame_idx = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(round(frame_idx * frame_period_ms))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            bbox = None
            if result.face_landmarks:
                bbox = bbox_from_lips(result.face_landmarks[0], w, h, LIPS_IDX, PAD_FRAC)

            if bbox is None:
                bbox = prev_bbox if prev_bbox is not None else fallback_bbox(w, h)
            prev_bbox = bbox

            x0, y0, x1, y1 = bbox
            mouth = frame_bgr[y0:y1, x0:x1]
            mouth = cv2.resize(mouth, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
            writer.write(mouth)

            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        landmarker.close()

def main():
    ensure_ffmpeg()
    model_path = ensure_landmarker_model()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MOUTH_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TXT_DIR.mkdir(parents=True, exist_ok=True)

    movs = [DOWNLOADS / "IMG_7713.MOV"]
    if not movs:
        print("No .mov files found in Downloads")
        return

    print(f"Found {len(movs)} .mov files")

    for src in movs:
        stem = src.stem
        raw_copy = RAW_DIR / src.name
        mouth_out = MOUTH_DIR / f"{stem}_mouth.mp4"
        wav_out = AUDIO_DIR / f"{stem}.wav"
        txt_out = TXT_DIR / f"{stem}.txt"

        print(f"\nProcessing {src.name}")

        if not raw_copy.exists():
            shutil.copy2(src, raw_copy)

        crop_mouth_video(raw_copy, mouth_out, model_path)
        extract_audio_ffmpeg(raw_copy, wav_out)

        if not txt_out.exists():
            txt_out.write_text("replace this with the spoken text\n", encoding="utf-8")

        print(f"  raw   -> {raw_copy}")
        print(f"  mouth -> {mouth_out}")
        print(f"  wav   -> {wav_out}")
        print(f"  txt   -> {txt_out}")

    print(f"\nDone. Outputs are in: {OUT_ROOT}")

if __name__ == "__main__":
    main()