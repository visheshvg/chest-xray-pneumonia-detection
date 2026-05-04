"""Run all models end-to-end and produce the comparison table."""
import os
import sys
import random
import numpy as np
import torch

import config
from data_loader import (
    fix_validation_split,
    compute_class_weights,
    get_data_loaders,
)
from models.custom_cnn import build_custom_cnn
from models.pretrained import build_pretrained_model, build_mobilenetv4, unfreeze_for_finetuning
from train import train_model
from evaluate import (
    plot_training_curves,
    evaluate_model,
    build_comparison_table,
)


class Tee:
    """Write to both the console and a log file simultaneously."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_keras_style(name, build_fn, img_size, two_phase, batch_size,
                    train_df, val_df, test_df, pos_weight):
    print("\n" + "=" * 60)
    print(f"Running {name}  (device: {config.DEVICE})")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_data_loaders(
        train_df, val_df, test_df, img_size, batch_size=batch_size,
    )

    model = build_fn()
    model, history = train_model(
        model, name, train_loader, val_loader, pos_weight,
        two_phase=two_phase,
        unfreeze_fn=unfreeze_for_finetuning if two_phase else None,
    )
    plot_training_curves(history, name)
    return evaluate_model(model, test_loader, name)


def run_mobilenetv4(train_df, val_df, test_df, pos_weight):
    """MobileNetV4 (Google, ECCV 2024) — advanced CNN via timm."""
    print("\n" + "=" * 60)
    print(f"Running mobilenetv4  (device: {config.DEVICE})")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_data_loaders(
        train_df, val_df, test_df, img_size=224, batch_size=config.BATCH_SIZE,
    )
    model = build_mobilenetv4(num_classes=1)
    model, history = train_model(
        model, "mobilenetv4", train_loader, val_loader, pos_weight,
        two_phase=False,
    )
    plot_training_curves(history, "mobilenetv4")
    return evaluate_model(model, test_loader, "mobilenetv4")


if __name__ == "__main__":
    log_path = os.path.join(config.RESULTS_DIR, "training_log.txt")
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    print(f"Logging to {log_path}")

    set_seed(config.RANDOM_SEED)
    print(f"Device: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Prepare data once
    train_df, val_df, test_df = fix_validation_split()
    pos_weight = compute_class_weights(train_df)

    # 2. Define runs: (name, build_fn, img_size, two_phase, batch_size)
    runs = [
        ("custom_cnn",
         lambda: build_custom_cnn(num_classes=1),
         config.IMG_SIZE_DEFAULT, False, config.BATCH_SIZE),

        ("densenet121",
         lambda: build_pretrained_model("densenet121", num_classes=1),
         config.IMG_SIZE_DEFAULT, True, config.BATCH_SIZE),

        ("resnet50",
         lambda: build_pretrained_model("resnet50", num_classes=1),
         config.IMG_SIZE_DEFAULT, True, config.BATCH_SIZE),

        ("efficientnetb4",
         lambda: build_pretrained_model("efficientnetb4", num_classes=1),
         config.IMG_SIZE_EFFICIENTNET_B4, True, config.BATCH_SIZE_B4),
    ]

    results = {}
    for name, build_fn, img_size, two_phase, bs in runs:
        try:
            results[name] = run_keras_style(
                name, build_fn, img_size, two_phase, bs,
                train_df, val_df, test_df, pos_weight,
            )
        except Exception as e:
            print(f"!! {name} failed: {e}")

    # 3. MobileNetV4
    try:
        results["mobilenetv4"] = run_mobilenetv4(train_df, val_df, test_df, pos_weight)
    except Exception as e:
        print(f"!! mobilenetv4 failed: {e}")

    # 4. Final comparison table
    build_comparison_table(results)