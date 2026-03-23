import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


INPUT_ROOT = Path("/Users/jasonherrmann/VSRProjLocal/PhonemeSegments")
OUTPUT_ROOT = Path("/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments24x24x9")

T = 9
H = 24
W = 24


# mp4

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(str(video_path))

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (W, H))

        frames.append(gray)

    cap.release()

    if len(frames) == 0:
        return

    # enforce exactly 9 frames
    if len(frames) > T:
        start = (len(frames) - T) // 2
        frames = frames[start:start+T]

    while len(frames) < T:
        frames.append(frames[-1])

    arr = np.stack(frames, axis=0)
    arr = np.expand_dims(arr, axis=-1).astype(np.uint8)

    np.save(output_path, arr)


# main

def main():

    mp4_files = list(INPUT_ROOT.rglob("*.mp4"))

    print("Total videos:", len(mp4_files))

    for mp4_path in tqdm(mp4_files):

        relative = mp4_path.relative_to(INPUT_ROOT)
        out_path = OUTPUT_ROOT / relative

        out_path = out_path.with_suffix(".npy")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            continue

        process_video(mp4_path, out_path)


if __name__ == "__main__":
    main()