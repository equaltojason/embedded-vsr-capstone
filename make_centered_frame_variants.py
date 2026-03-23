#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations


#Sources
#- /Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments32x32x9
#- /Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments48x48x9

#Generated datasets:
#- /Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments32x32x7
#- /Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments32x32x5
#- /Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments48x48x7
#- /Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments48x48x5

#cropping rule:
#7: remove 1 from front and 1 from back
#5: remove 2 frames from front and 2 from back

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

DATASET_ROOTS: Dict[str, Path] = {
    "32x32x9": Path("/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments32x32x9"),
    "48x48x9": Path("/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments48x48x9"),
}

TARGET_SPECS: Dict[int, Tuple[int, int]] = {
    7: (1, 8),# frames [1:8]
    5: (2, 7),#frames [2:7]
}


def shape_to_hwt(arr: np.ndarray, h: int, w: int, t: int = 9) -> np.ndarray:
 
    #Convert common saved array layouts into (H, W, T).
    #(H, W, T)
    #(T, H, W)
    #H, W, T, 1)
    #(T, H, W, 1)
    arr = np.asarray(arr)

    if arr.ndim == 3:
        if arr.shape == (h, w, t):
            return arr
        if arr.shape == (t, h, w):
            return np.transpose(arr, (1, 2, 0))
        raise ValueError(f"Unsupported 3D shape: {arr.shape}")

    if arr.ndim == 4:
        if arr.shape == (h, w, t, 1):
            return arr[..., 0]
        if arr.shape == (t, h, w, 1):
            return np.transpose(arr[..., 0], (1, 2, 0))
        raise ValueError(f"Unsupported 4D shape: {arr.shape}")

    raise ValueError(f"Unsupported rank {arr.ndim} for shape {arr.shape}")


def convert_source_dataset(
    src_root: Path,
    dst_root: Path,
    h: int,
    w: int,
    target_t: int,
    overwrite: bool
) -> None:
    keep_start, keep_end = TARGET_SPECS[target_t]

    files = list(src_root.rglob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy files found in {src_root}")

    print(f"\nConverting {src_root.name} -> {dst_root.name}")
    print(f"Files found: {len(files)}")

    for src_file in tqdm(files, desc=f"{src_root.name} -> {target_t}f"):
        rel = src_file.relative_to(src_root)
        dst_file = dst_root / rel

        if dst_file.exists() and not overwrite:
            continue

        arr = np.load(src_file)
        arr = shape_to_hwt(arr, h=h, w=w, t=9)
        cropped = arr[:, :, keep_start:keep_end]

        dst_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst_file, cropped)


def infer_hw_from_name(name: str) -> Tuple[int, int]:
    if name.startswith("32x32x9"):
        return 32, 32
    if name.startswith("48x48x9"):
        return 48, 48
    raise ValueError(f"Unsupported source dataset name: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create centered 5-frame and 7-frame datasets from 9-frame .npy clips"
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=list(DATASET_ROOTS.keys()),
        choices=list(DATASET_ROOTS.keys()),
        help="Source 9-frame datasets to convert.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        type=int,
        default=[7, 5],
        choices=[7, 5],
        help="Target frame counts to create.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    for source_name in args.sources:
        src_root = DATASET_ROOTS[source_name]
        h, w = infer_hw_from_name(source_name)

        if not src_root.exists():
            print(f"Skipping missing source: {src_root}")
            continue

        for target_t in args.targets:
            dst_root = src_root.parent / src_root.name.replace("x9", f"x{target_t}")
            convert_source_dataset(
                src_root=src_root,
                dst_root=dst_root,
                h=h,
                w=w,
                target_t=target_t,
                overwrite=args.overwrite,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
