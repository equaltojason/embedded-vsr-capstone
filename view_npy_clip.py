import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def to_display(frame: np.ndarray) -> np.ndarray:
    #if stored as int8, shift back to grayscale.
    if frame.dtype == np.int8:
        return np.clip(frame.astype(np.int16) + 128, 0, 255).astype(np.uint8)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", help="Path to .npy clip")
    parser.add_argument("--outdir", default=None, help="Folder to save PNGs")
    args = parser.parse_args()

    npy_path = Path(args.npy_path).expanduser()
    x = np.load(npy_path)

    print("shape:", x.shape)
    print("dtype:", x.dtype)
    print("min/max:", x.min(), x.max())

    if x.shape != (32, 32, 7):
        raise ValueError(f"Expected shape (32, 32, 7), got {x.shape}")

    outdir = Path(args.outdir).expanduser() if args.outdir else npy_path.parent / f"{npy_path.stem}_frames"
    outdir.mkdir(parents=True, exist_ok=True)

    #save individual frames
    for i in range(7):
        frame = to_display(x[:, :, i])
        plt.imsave(outdir / f"{npy_path.stem}_frame_{i}.png", frame, cmap="gray", vmin=0, vmax=255)

    #show 7-frame strip
    fig, axes = plt.subplots(1, 7, figsize=(14, 2.5))
    for i in range(7):
        frame = to_display(x[:, :, i])
        axes[i].imshow(frame, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(f"F{i+1}")
        axes[i].axis("off")

    plt.tight_layout()

    strip_path = outdir / f"{npy_path.stem}_strip.png"
    plt.savefig(strip_path, dpi=200, bbox_inches="tight")
    print("Saved PNGs to:", outdir)
    print("Saved strip to:", strip_path)

    plt.show()


if __name__ == "__main__":
    main()