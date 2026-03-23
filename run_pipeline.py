import os, sys, json, shutil, subprocess, time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from concurrent.futures import ThreadPoolExecutor, as_completed



OUT_W = OUT_H = 96
PAD_FRAC = 0.35

LIPS_IDX = sorted(set([
    0, 13, 14, 37, 39, 40, 267, 269, 270, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 95, 88, 178, 87, 317, 402, 318, 324
]))

MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/face_landmarker.task"


def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("ERROR: ffmpeg not found. Install with: brew install ffmpeg")
        sys.exit(1)

def cache_dir() -> Path:
    d = Path.home() / ".cache" / "vsr_pipeline"
    d.mkdir(parents=True, exist_ok=True)
    return d

def ensure_landmarker_model() -> Path:
    model_path = cache_dir() / "face_landmarker.task"
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    print(f"Downloading FaceLandmarker model to {model_path} ...")
    subprocess.run(["curl", "-L", "-o", str(model_path), MODEL_URL], check=True)
    return model_path

def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024**3)

def fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600); seconds -= 3600*h
    m = int(seconds // 60); seconds -= 60*m
    s = int(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}"

def file_ok(path: Path, min_bytes: int = 1000) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def align_to_transcript(path: Path) -> str:
    words = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        token = parts[2].strip().lower()
        if token == "sil":
            continue
        words.append(token)
    return " ".join(words)


def extract_audio_ffmpeg(video_path: Path, wav_path: Path) -> bool:
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
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        if wav_path.exists():
            try: wav_path.unlink()
            except: pass
        return False


def make_landmarker(model_path: Path):
    BaseOptions = python.BaseOptions
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1
    )
    return FaceLandmarker.create_from_options(options)

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
    x1 = int(np.ceil (x1 + pad_x))
    y0 = int(np.floor(y0 - pad_y))
    y1 = int(np.ceil (y1 + pad_y))

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

def crop_mouth_video(video_path: Path, out_path: Path, model_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

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
    ok = True

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
                landmarks = result.face_landmarks[0]
                bbox = bbox_from_lips(landmarks, w, h, LIPS_IDX, PAD_FRAC)

            if bbox is None:
                bbox = prev_bbox if prev_bbox is not None else fallback_bbox(w, h)
            prev_bbox = bbox

            x0, y0, x1, y1 = bbox
            mouth = frame_bgr[y0:y1, x0:x1]
            mouth = cv2.resize(mouth, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
            writer.write(mouth)

            frame_idx += 1
    except Exception:
        ok = False
    finally:
        cap.release()
        writer.release()
        landmarker.close()

    if not ok:
        if out_path.exists():
            try: out_path.unlink()
            except: pass
    return ok


def run_audio_batch(pairs: List[Tuple[Path, Path]], max_workers: int, failures: List[dict], split: str):
    if max_workers <= 1:
        for vp, wav_out in pairs:
            ok = extract_audio_ffmpeg(vp, wav_out)
            if not ok:
                failures.append({"split": split, "file": vp.name, "stage": "audio", "reason": "ffmpeg failed"})
        return

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(extract_audio_ffmpeg, vp, wav_out): (vp, wav_out) for vp, wav_out in pairs}
        for fut in as_completed(futs):
            vp, _ = futs[fut]
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if not ok:
                failures.append({"split": split, "file": vp.name, "stage": "audio", "reason": "ffmpeg failed"})

def run_text_batch(pairs: List[Tuple[Path, Path]], max_workers: int, failures: List[dict], split: str):
    def one(ap: Path, txt_out: Path) -> bool:
        try:
            transcript = align_to_transcript(ap)
            safe_write_text(txt_out, (transcript + "\n") if transcript.strip() else "")
            return True
        except Exception:
            return False

    if max_workers <= 1:
        for ap, txt_out in pairs:
            ok = one(ap, txt_out)
            if not ok:
                failures.append({"split": split, "file": ap.name, "stage": "text", "reason": "align parse/write failed"})
        return

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(one, ap, txt_out): (ap, txt_out) for ap, txt_out in pairs}
        for fut in as_completed(futs):
            ap, _ = futs[fut]
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if not ok:
                failures.append({"split": split, "file": ap.name, "stage": "text", "reason": "align parse/write failed"})


def verify_split_outputs(split: str, expected: int, out_root: Path, strict: bool = True) -> bool:
    mouth_dir = out_root / "CORPUSprocessed" / split
    audio_dir = out_root / "CORPUSaudio" / split
    txt_dir   = out_root / "CORPUStxt" / split

    mouth_n = len(list(mouth_dir.glob("*.mp4"))) if mouth_dir.exists() else 0
    audio_n = len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0
    txt_n   = len(list(txt_dir.glob("*.txt"))) if txt_dir.exists() else 0

    ok = (mouth_n >= expected and audio_n >= expected and txt_n >= expected)
    if strict:
        ok = (mouth_n == expected and audio_n == expected and txt_n == expected)

    status = "OK" if ok else "MISMATCH"
    print(f"[VERIFY:{status}] {split}: expected={expected} mouth={mouth_n} wav={audio_n} txt={txt_n}")
    return ok

def process_split(
    split_dir: Path,
    out_root: Path,
    min_free_gb: float,
    failures: list,
    limit_per_split: Optional[int],
    audio_workers: int,
    text_workers: int,
    verify_expected: Optional[int],
    strict_verify: bool,
    overall_progress: Optional[tqdm] = None,
):
    split = split_dir.name
    print(f"\n=== Split: {split} ===")

    in_videos = split_dir
    in_align  = split_dir / "align"

    out_mouth = out_root / "CORPUSprocessed" / split
    out_audio = out_root / "CORPUSaudio" / split
    out_txt   = out_root / "CORPUStxt" / split

    out_mouth.mkdir(parents=True, exist_ok=True)
    out_audio.mkdir(parents=True, exist_ok=True)
    out_txt.mkdir(parents=True, exist_ok=True)

    # Discover videos + alignments
    videos = sorted([
        p for p in in_videos.iterdir()
        if p.is_file() and p.suffix.lower() in (".mpg", ".mpeg")
    ])
    if not videos:
        print(f"[WARN] No videos found in {in_videos}")
        return 0

    if not in_align.exists():
        print(f"[WARN] Missing align dir: {in_align} (skipping split)")
        return 0

    align_map = {p.stem: p for p in in_align.glob("*.align")}
    paired_all = [vp for vp in videos if vp.stem in align_map]
    missing_align = len(videos) - len(paired_all)

    print(f"Videos: {len(videos)} | Paired with .align: {len(paired_all)} | Missing align: {missing_align}")
    if missing_align > 0:
        print("[WARN] Some videos are missing .align; they will be skipped.")

    # (not fully processed) files first
    def is_done(vp: Path) -> bool:
        stem = vp.stem
        mouth_out = out_mouth / f"{stem}_mouth.mp4"
        wav_out   = out_audio / f"{stem}.wav"
        txt_out   = out_txt   / f"{stem}.txt"
        return file_ok(mouth_out, 1000) and file_ok(wav_out, 1000) and txt_out.exists()

    paired_todo = [vp for vp in paired_all if not is_done(vp)]
    paired_done = [vp for vp in paired_all if is_done(vp)]

    if paired_todo:
        paired = paired_todo
        print(f"[INFO] Todo: {len(paired_todo)} | Already done: {len(paired_done)} (will process todo first)")
    else:
        paired = paired_all
        print("[INFO] Everything appears done already; using full paired list (will mostly skip).")

    # test limit
    if limit_per_split is not None:
        paired = paired[:limit_per_split]

    # expected
    expected = verify_expected if verify_expected is not None else len(paired)

    model_path = ensure_landmarker_model()

   #stage 1
    split_start = time.time()
    total = len(paired)
    done = 0

    pbar = tqdm(total=total, desc=f"{split} mouth", leave=False)
    for vp in paired:
        if free_gb(out_root) < min_free_gb:
            raise RuntimeError(f"Low disk space: free < {min_free_gb} GB. Stopping safely.")

        stem = vp.stem
        mouth_out = out_mouth / f"{stem}_mouth.mp4"

        if not file_ok(mouth_out, 1000):
            ok = crop_mouth_video(vp, mouth_out, model_path)
            if not ok:
                failures.append({"split": split, "file": vp.name, "stage": "mouth", "reason": "crop failed"})
                # continue to next file

        done += 1
        pbar.update(1)

        #ETA
        elapsed = time.time() - split_start
        rate = done / max(1e-9, elapsed)
        eta = (total - done) / max(1e-9, rate)
        pbar.set_postfix_str(f"elapsed={fmt_hms(elapsed)} eta={fmt_hms(eta)}")

        if overall_progress is not None:
            overall_progress.update(1)

    pbar.close()

    #Stage 2
    audio_jobs = []
    for vp in paired:
        wav_out = out_audio / f"{vp.stem}.wav"
        if not file_ok(wav_out, 1000):
            audio_jobs.append((vp, wav_out))

    if audio_jobs:
        print(f"[AUDIO] Extracting {len(audio_jobs)} wav files (workers={audio_workers}) ...")
        run_audio_batch(audio_jobs, max_workers=audio_workers, failures=failures, split=split)
    else:
        print("[AUDIO] All wav files already present; skipping.")

    #Stage 3: Text
    text_jobs = []
    for vp in paired:
        ap = align_map[vp.stem]
        txt_out = out_txt / f"{vp.stem}.txt"
        if not txt_out.exists():
            text_jobs.append((ap, txt_out))

    if text_jobs:
        print(f"[TEXT] Writing {len(text_jobs)} transcripts (workers={text_workers}) ...")
        run_text_batch(text_jobs, max_workers=text_workers, failures=failures, split=split)
    else:
        print("[TEXT] All transcripts already present; skipping.")

    # Verify
    if verify_expected is not None or strict_verify:
        verify_split_outputs(split, expected=expected, out_root=out_root, strict=strict_verify)

    return total


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in-root", type=str, default=str(Path.home() / "Downloads" / "data-2"))
    p.add_argument("--out-root", type=str, default=str(Path.home() / "VSRProjLocal"))
    p.add_argument("--min-free-gb", type=float, default=20.0)
    p.add_argument("--splits", type=str, default="auto", help="auto or comma-separated split names")

    # New:
    p.add_argument("--limit-per-split", type=int, default=None, help="Process only first N paired files per split (testing).")
    p.add_argument("--audio-workers", type=int, default=4, help="Parallel workers for audio extraction (safe).")
    p.add_argument("--text-workers", type=int, default=4, help="Parallel workers for text extraction (safe).")
    p.add_argument("--verify-expected", type=int, default=None, help="If set, verify each split has exactly this many outputs (e.g., 1000).")
    p.add_argument("--strict-verify", action="store_true", help="If set, require exact match (not >=) when verifying.")
    args = p.parse_args()

    ensure_ffmpeg()

    in_root = Path(args.in_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.splits == "auto":
        split_dirs = sorted([p for p in in_root.iterdir()
                             if p.is_dir() and p.name.startswith("s") and p.name.endswith("_processed")])
    else:
        names = [s.strip() for s in args.splits.split(",") if s.strip()]
        split_dirs = [in_root / n for n in names]

    # Estimate overall 
    total_overall = 0
    for sd in split_dirs:
        if not sd.exists():
            continue
        vids = [p for p in sd.iterdir() if p.is_file() and p.suffix.lower() in (".mpg", ".mpeg")]
        align_dir = sd / "align"
        align_map = {p.stem for p in align_dir.glob("*.align")} if align_dir.exists() else set()
        paired = [vp for vp in vids if vp.stem in align_map]
        if args.limit_per_split is not None:
            paired = paired[:args.limit_per_split]
        total_overall += len(paired)

    failures = []
    start = time.time()

    overall = tqdm(total=total_overall, desc="OVERALL mouth progress", leave=True) if total_overall > 0 else None

    for sd in split_dirs:
        if not sd.exists():
            print(f"[WARN] Missing split dir: {sd}")
            continue
        try:
            process_split(
                sd,
                out_root=out_root,
                min_free_gb=args.min_free_gb,
                failures=failures,
                limit_per_split=args.limit_per_split,
                audio_workers=max(1, args.audio_workers),
                text_workers=max(1, args.text_workers),
                verify_expected=args.verify_expected,
                strict_verify=args.strict_verify,
                overall_progress=overall,
            )
        except Exception as e:
            failures.append({"split": sd.name, "file": None, "stage": "split", "reason": str(e)})

    if overall is not None:
        overall.close()

    elapsed = time.time() - start
    report_dir = out_root / "_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"failures_{int(time.time())}.json"
    report_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"Elapsed: {elapsed/3600:.2f} hours ({fmt_hms(elapsed)})")
    print(f"Failures: {len(failures)}")
    print(f"Failure report: {report_path}")

    # varify with global pass
    if args.verify_expected is not None:
        print("\n=== FINAL VERIFY (all splits) ===")
        ok_all = True
        for sd in split_dirs:
            if not sd.exists():
                continue
            ok = verify_split_outputs(sd.name, expected=args.verify_expected, out_root=out_root, strict=args.strict_verify)
            ok_all = ok_all and ok
        print("FINAL VERIFY:", "OK" if ok_all else "MISMATCHES FOUND")


if __name__ == "__main__":
    main()