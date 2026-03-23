from __future__ import annotations

import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from praatio import textgrid

CLIP_LEN = 9
HALF = CLIP_LEN // 2
IGNORE = {"sil", "sp", "spn", "silence", "", "pau"}
SKIP_IGNORE = True
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

def safe_name(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")

def normalize_stem(stem: str) -> str:
    for suf in ["_mouth", "_roi", "_crop", "_cropped", "_processed"]:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem

def build_video_map(vid_dir: Path) -> dict[str, Path]:
    m: dict[str, Path] = {}
    for ext in ["*.mp4", "*.mpg", "*.avi", "*.mov", "*.mkv"]:
        for p in vid_dir.glob(ext):
            base = normalize_stem(p.stem)
            if base not in m:
                m[base] = p
    return m

def get_phone_intervals(tg_path: Path):
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)

    tier_name = None
    for cand in ["phones", "phone", "phonemes", "phoneme"]:
        if cand in tg.tierNames:
            tier_name = cand
            break
    if tier_name is None:
        tier_name = tg.tierNames[0]

    tier = tg.getTier(tier_name)
    intervals = []
    for (start, end, label) in tier.entries:
        lab = (label or "").strip().lower()
        if SKIP_IGNORE and (lab in IGNORE):
            continue
        intervals.append((float(start), float(end), lab))
    return intervals, tier_name

def get_video_fps_and_count(cap: cv2.VideoCapture):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, n

def read_frame_at(cap: cv2.VideoCapture, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return ok, frame

def write_clip_mp4(out_path: Path, frames: list, fps: float):
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(out_path), FOURCC, fps, (w, h))
    for fr in frames:
        out.write(fr)
    out.release()

def segment_split(
    split: str,
    tg_dir: Path,
    vid_dir: Path,
    out_root: Path,
):
    out_root.mkdir(parents=True, exist_ok=True)

    tg_files = sorted(tg_dir.glob("*.TextGrid"))
    if not tg_files:
        raise FileNotFoundError(f"No TextGrid files found in {tg_dir}")

    video_map = build_video_map(vid_dir)

    manifest_rows = []
    saved = 0
    skipped_no_video = 0
    skipped_bounds = 0
    skipped_readfail = 0

    for tg_path in tqdm(tg_files, desc=f"{split} phoneme segmenting"):
        stem = tg_path.stem
        vid_path = video_map.get(stem)
        if vid_path is None:
            skipped_no_video += 1
            continue

        intervals, tier_name = get_phone_intervals(tg_path)
        if not intervals:
            continue

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            skipped_no_video += 1
            continue

        fps, n = get_video_fps_and_count(cap)
        if n < CLIP_LEN:
            cap.release()
            continue

        for idx, (start, end, phone) in enumerate(intervals):
            center_t = 0.5 * (start + end)
            k = int(round(center_t * fps))
            a = k - HALF
            b = k + HALF

            if a < 0 or b >= n:
                skipped_bounds += 1
                continue

            phone_dir = out_root / safe_name(phone)
            out_name = f"{stem}_{idx:03d}_{safe_name(phone)}_{a:05d}-{b:05d}.mp4"
            out_path = phone_dir / out_name

            # resume-safe
            if out_path.exists() and out_path.stat().st_size > 1000:
                continue

            frames = []
            ok_all = True
            for fi in range(a, b + 1):
                ok, fr = read_frame_at(cap, fi)
                if not ok or fr is None:
                    ok_all = False
                    break
                frames.append(fr)

            if not ok_all or len(frames) != CLIP_LEN:
                skipped_readfail += 1
                continue

            write_clip_mp4(out_path, frames, fps)
            saved += 1

            manifest_rows.append({
                "split": split,
                "stem": stem,
                "video": str(vid_path),
                "textgrid": str(tg_path),
                "tier": tier_name,
                "phone": phone,
                "start_s": start,
                "end_s": end,
                "center_s": center_t,
                "fps": fps,
                "frame_center": k,
                "frame_start": a,
                "frame_end": b,
                "out_path": str(out_path),
            })

        cap.release()

    manifest_path = out_root / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print("\nDone.")
    print("Saved clips:", saved)
    print("Skipped (no matching video):", skipped_no_video)
    print("Skipped (out-of-bounds window):", skipped_bounds)
    print("Skipped (frame read fail):", skipped_readfail)
    print("Manifest:", manifest_path)
    print("Output root:", out_root)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, help="e.g. s1_processed")
    p.add_argument("--root", default=str(Path.home() / "VSRProjLocal"))
    args = p.parse_args()

    root = Path(args.root).expanduser()
    split = args.split

    tg_dir  = root / "CORPUSalignment" / split
    vid_dir = root / "CORPUSprocessed" / split
    out_dir = root / "PhonemeSegments" / split

    print("TextGrid dir:", tg_dir)
    print("Video dir:", vid_dir)
    print("Out dir:", out_dir)

    segment_split(split=split, tg_dir=tg_dir, vid_dir=vid_dir, out_root=out_dir)

if __name__ == "__main__":
    main()