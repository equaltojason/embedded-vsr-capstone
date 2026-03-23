import argparse
from pathlib import Path

import cv2
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to cropped mouth mp4")
    p.add_argument("--start", type=float, required=True, help="Segment start time in seconds")
    p.add_argument("--end", type=float, required=True, help="Segment end time in seconds")
    p.add_argument("--out", required=True, help="Output .npy path")
    args = p.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seg_start = max(0, int(round(args.start * fps)))
    seg_end = min(total_frames - 1, int(round(args.end * fps)))

    if seg_end <= seg_start:
        raise ValueError("Bad segment bounds")

    idxs = np.linspace(seg_start, seg_end, 7, dtype=int)
    frames = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {idx}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

        # Temporary int8 mapping centered around zero.
        q = small.astype(np.int16) - 128
        q = np.clip(q, -128, 127).astype(np.int8)

        frames.append(q)

    cap.release()

    clip = np.stack(frames, axis=-1)  # (32, 32, 7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, clip)

    print("Saved:", out_path)
    print("Shape:", clip.shape)
    print("Dtype:", clip.dtype)
    print("Min/Max:", clip.min(), clip.max())

if __name__ == "__main__":
    main()