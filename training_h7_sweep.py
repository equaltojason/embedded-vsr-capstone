#!/usr/bin/env python3
#coding: utf-8

from __future__ import annotations


#Run small/medium/large 2D CNN sweepsfor a chosen dataset.

#Supported datasets:
#32x32x5, 32x32x7, 32x32x9
#48x48x5, 48x48x7, 48x48x9

#Uses speaker splitting and classes:
#M, SH, O, E, FV

#Outputs per-model artifacts plus:
#sweep_summary.csv
#sweep_accuracy.png


import argparse
import csv
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


@dataclass
class DatasetConfig:
    name: str
    data_root: str
    h: int
    w: int
    t: int


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "24x24x5": DatasetConfig("24x24x5", "/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments24x24x5", 24, 24, 5),
    "24x24x7": DatasetConfig("24x24x7", "/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments24x24x7", 24, 24, 7),
    "24x24x9": DatasetConfig("24x24x9", "/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments24x24x9", 24, 24, 9),
    "28x28x5": DatasetConfig("28x28x5", "/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments28x28x5", 28, 28, 5),
    "28x28x7": DatasetConfig("28x28x7", "/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments28x28x7", 28, 28, 7),
    "28x28x9": DatasetConfig("28x28x9", "/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments28x28x9", 28, 28, 9),
    
    "32x32x5": DatasetConfig("32x32x5", "/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments32x32x5", 32, 32, 5),
    "32x32x7": DatasetConfig("32x32x7", "/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments32x32x7", 32, 32, 7),
    "32x32x9": DatasetConfig("32x32x9", "/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments32x32x9", 32, 32, 9),
    "48x48x5": DatasetConfig("48x48x5", "/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments48x48x5", 48, 48, 5),
    "48x48x7": DatasetConfig("48x48x7", "/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments48x48x7", 48, 48, 7),
    "48x48x9": DatasetConfig("48x48x9", "/Users/jasonherrmann/VSRProjLocal/SplitPhonemeSegments48x48x9", 48, 48, 9),
}

DEFAULT_OUTPUT_ROOT = Path("/Users/jasonherrmann/VSRProj/artifacts")
DEFAULT_SEED = 1337
DEFAULT_VAL_FRAC = 0.10
DEFAULT_TEST_FRAC = 0.10
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 8
DEFAULT_LR = 1e-3

CLASS_NAMES = ["M", "SH", "O", "E", "FV"]
NUM_CLASSES = len(CLASS_NAMES)

#viseme groups
BILABIALS = {"b", "bʲ", "p", "pʰ", "pʲ", "m"}
FRICATIVES = {"s", "z", "tʃ", "dʒ"}
ROUND = {"ow", "ʉ", "ʉː"}
SPREAD = {"i", "iː", "ɪ"}
LABIODENTALS = {"f", "v", "vʲ"}

LABEL_MAP: Dict[str, int] = {}
for tok in BILABIALS:
    LABEL_MAP[tok] = 0
for tok in FRICATIVES:
    LABEL_MAP[tok] = 1
for tok in ROUND:
    LABEL_MAP[tok] = 2
for tok in SPREAD:
    LABEL_MAP[tok] = 3
for tok in LABIODENTALS:
    LABEL_MAP[tok] = 4

spk_re = re.compile(r"^s(\d+)_processed$")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential small/medium/large sweep for H7-friendly VSR models")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS.keys()), required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-train-per-class", type=int, default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=["small", "medium", "large"],
        default=["small", "medium", "large"],
    )
    parser.add_argument("--no-int8-export", action="store_true")
    return parser.parse_args()


def normalize_token(token: str) -> str:
    return token.strip().lower().replace(" ", "")


def get_speaker_dirs(data_root: Path) -> List[Path]:
    speaker_dirs = [d for d in data_root.iterdir() if d.is_dir() and spk_re.match(d.name)]
    speaker_dirs.sort(key=lambda p: int(spk_re.match(p.name).group(1)))
    return speaker_dirs


def infer_label_token_from_npy(npy_path: Path) -> str:
    return normalize_token(npy_path.parent.name)


def find_samples(data_root: Path) -> List[Tuple[str, str, str, int]]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    speaker_dirs = get_speaker_dirs(data_root)
    if not speaker_dirs:
        raise RuntimeError(f"No speaker dirs found in {data_root}")

    found: List[Tuple[str, str, str, int]] = []
    total_npy = 0
    total_mapped = 0
    total_unmapped = 0

    for spk_dir in tqdm(speaker_dirs, desc="Scanning speakers"):
        spk_id = spk_dir.name.replace("_processed", "")
        for npy_path in spk_dir.rglob("*.npy"):
            total_npy += 1
            token = infer_label_token_from_npy(npy_path)
            if token not in LABEL_MAP:
                total_unmapped += 1
                continue
            y = LABEL_MAP[token]
            found.append((str(npy_path), spk_id, token, y))
            total_mapped += 1

    print(f"\nTotal .npy files found: {total_npy}")
    print(f"Mapped usable samples: {total_mapped}")
    print(f"Unmapped/ignored samples: {total_unmapped}")
    return found


def class_count_dict(entries: Sequence[Tuple[str, str, str, int]]) -> Dict[str, int]:
    counts = {i: 0 for i in range(NUM_CLASSES)}
    for _, _, _, y in entries:
        counts[y] += 1
    return {CLASS_NAMES[k]: v for k, v in counts.items()}


def cap_entries_per_class(
    entries: Sequence[Tuple[str, str, str, int]],
    max_per_class: int,
    seed: int
) -> List[Tuple[str, str, str, int]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[Tuple[str, str, str, int]]] = {i: [] for i in range(NUM_CLASSES)}
    for e in entries:
        by_class[e[3]].append(e)

    capped: List[Tuple[str, str, str, int]] = []
    for y in range(NUM_CLASSES):
        bucket = by_class[y][:]
        rng.shuffle(bucket)
        capped.extend(bucket[:max_per_class])

    rng.shuffle(capped)
    return capped


def shape_to_hwc(arr: np.ndarray, h: int, w: int, t: int) -> np.ndarray:
    """
    Convert common saved array layouts into (H, W, T).
    Supported:
    - (H, W, T)
    - (T, H, W)
    - (H, W, T, 1)
    - (T, H, W, 1)
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        if arr.shape == (h, w, t):
            pass
        elif arr.shape == (t, h, w):
            arr = np.transpose(arr, (1, 2, 0))
        else:
            raise ValueError(f"Unsupported 3D array shape: {arr.shape}")
    elif arr.ndim == 4:
        if arr.shape == (h, w, t, 1):
            arr = arr[..., 0]
        elif arr.shape == (t, h, w, 1):
            arr = np.transpose(arr[..., 0], (1, 2, 0))
        else:
            raise ValueError(f"Unsupported 4D array shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported array rank {arr.ndim} for shape {arr.shape}")

    if arr.shape[:2] != (h, w):
        raise ValueError(f"Spatial mismatch: got {arr.shape}, expected ({h},{w},*)")

    if arr.shape[2] > t:
        start = (arr.shape[2] - t) // 2
        arr = arr[:, :, start:start + t]
    elif arr.shape[2] < t:
        last = arr[:, :, -1:]
        pad = np.repeat(last, t - arr.shape[2], axis=2)
        arr = np.concatenate([arr, pad], axis=2)

    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def preload_entries(entries, h: int, w: int, t: int, desc: str):
    X = []
    y = []
    for path, _, _, label in tqdm(entries, desc=desc):
        arr = np.load(path)
        arr = shape_to_hwc(arr, h=h, w=w, t=t)
        X.append(arr)
        y.append(label)
    X = np.asarray(X, dtype=np.float32)
    y = tf.keras.utils.to_categorical(np.asarray(y), NUM_CLASSES)
    return X, y


def build_model(input_shape, model_size: str) -> tf.keras.Model:
    filters_map = {
        "small": [16, 32, 48],
        "medium": [24, 48, 64],
        "large": [32, 64, 96],
    }
    f1, f2, f3 = filters_map[model_size]

    inp = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(f1, 3, padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(f2, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(f3, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return tf.keras.Model(inp, out)


def save_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(NUM_CLASSES), CLASS_NAMES)
    plt.yticks(range(NUM_CLASSES), CLASS_NAMES)
    plt.colorbar()
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def save_training_curves(history, out_path: Path) -> None:
    hist = history.history
    fig = plt.figure()
    plt.plot(hist.get("accuracy", []), label="train_acc")
    plt.plot(hist.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Curves")
    plt.legend()
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def run_one_model(
    model_size,
    cfg,
    output_dir,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    args
):
    model_dir = output_dir / model_size
    model_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(input_shape=(cfg.h, cfg.w, cfg.t), model_size=model_size)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_model.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True,
            monitor="val_accuracy",
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    save_training_curves(history, model_dir / "training_curves.png")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    (model_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (model_dir / "test_metrics.txt").write_text(
        f"test_loss={test_loss:.6f}\ntest_accuracy={test_acc:.6f}\n",
        encoding="utf-8"
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    save_confusion_matrix(cm, model_dir / "confusion_matrix.png")

    float_sm_dir = model_dir / "float_savedmodel"
    model.export(str(float_sm_dir))

    int8_ok = False
    if not args.no_int8_export:
        def representative_data_gen():
            for i in range(min(200, len(X_train))):
                yield [X_train[i:i+1]]

        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(str(float_sm_dir))
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            tflite_int8 = converter.convert()
            (model_dir / "model_int8.tflite").write_bytes(tflite_int8)
            int8_ok = True
        except Exception as exc:
            (model_dir / "tflite_export_error.txt").write_text(
                f"INT8 export failed: {exc}\n",
                encoding="utf-8"
            )

    return {
        "model_size": model_size,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "int8_export_ok": int(int8_ok),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = DATASET_CONFIGS[args.dataset]
    data_root = Path(args.data_root) if args.data_root else Path(cfg.data_root)

    suffix = f"sweep_{cfg.name}"
    if args.max_train_per_class is not None:
        suffix += f"_cap{args.max_train_per_class}pc"

    output_dir = Path(args.output_root) / suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== CONFIG ===")
    print(f"Dataset config      : {cfg.name}")
    print(f"Data root           : {data_root}")
    print(f"Output dir          : {output_dir}")
    print(f"Input shape         : {(cfg.h, cfg.w, cfg.t)}")
    print(f"Models              : {args.models}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Epochs              : {args.epochs}")
    print(f"Learning rate       : {args.lr}")
    print(f"Max train per class : {args.max_train_per_class}")

    entries = find_samples(data_root)
    if not entries:
        raise SystemExit("Found 0 usable samples after token mapping.")

    speakers = sorted({spk for _, spk, _, _ in entries}, key=lambda s: int(s[1:]))
    random.shuffle(speakers)

    n = len(speakers)
    n_test = max(1, int(args.test_frac * n))
    n_val = max(1, int(args.val_frac * n))
    if n_test + n_val >= n:
        raise ValueError("Not enough speakers for requested split.")

    test_speakers = set(speakers[:n_test])
    val_speakers = set(speakers[n_test:n_test + n_val])
    train_speakers = set(speakers[n_test + n_val:])

    def split_entries(ent):
        tr, va, te = [], [], []
        for e in ent:
            spk = e[1]
            if spk in test_speakers:
                te.append(e)
            elif spk in val_speakers:
                va.append(e)
            else:
                tr.append(e)
        return tr, va, te

    train_entries, val_entries, test_entries = split_entries(entries)

    print(
        f"\nSpeakers total={len(speakers)} "
        f"train={len(train_speakers)} val={len(val_speakers)} test={len(test_speakers)}"
    )
    print(f"Samples before caps train={len(train_entries)} val={len(val_entries)} test={len(test_entries)}")
    print("Train class counts before cap:", class_count_dict(train_entries))

    if args.max_train_per_class is not None:
        train_entries = cap_entries_per_class(train_entries, args.max_train_per_class, args.seed)
        print(f"\nApplied training cap of {args.max_train_per_class} per class.")
        print(f"Train samples after cap: {len(train_entries)}")
        print("Train class counts after cap:", class_count_dict(train_entries))

    print("\nPreloading train/val/test into RAM...")
    X_train, y_train = preload_entries(train_entries, cfg.h, cfg.w, cfg.t, "Preloading train")
    X_val, y_val = preload_entries(val_entries, cfg.h, cfg.w, cfg.t, "Preloading val")
    X_test, y_test = preload_entries(test_entries, cfg.h, cfg.w, cfg.t, "Preloading test")

    summary_rows = []
    for model_size in args.models:
        print(f"\n===== RUNNING MODEL: {model_size.upper()} =====")
        result = run_one_model(
            model_size=model_size,
            cfg=cfg,
            output_dir=output_dir,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            args=args,
        )
        summary_rows.append(result)

    csv_path = output_dir / "sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_size", "test_accuracy", "test_loss", "int8_export_ok"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    fig = plt.figure()
    xs = [row["model_size"] for row in summary_rows]
    ys = [row["test_accuracy"] for row in summary_rows]
    plt.plot(xs, ys, marker="o")
    plt.ylim(bottom=max(0.0, min(ys) - 0.05), top=min(1.0, max(ys) + 0.05))
    plt.xlabel("Model Size")
    plt.ylabel("Test Accuracy")
    plt.title(f"Sweep Accuracy: {cfg.name}")
    fig.tight_layout()
    plt.savefig(output_dir / "sweep_accuracy.png", dpi=200)
    plt.close(fig)

    run_config = {
        "dataset": asdict(cfg),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_train_per_class": args.max_train_per_class,
        "models": args.models,
        "class_names": CLASS_NAMES,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"\nDone. Artifacts written to: {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
