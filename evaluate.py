"""Plots, metrics, and the final comparison table."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
)

import config


@torch.no_grad()
def predict(model, loader):
    """Return (y_true, y_pred, y_probs) as numpy arrays."""
    model.eval().to(config.DEVICE)
    y_true, y_probs = [], []
    for imgs, labels in loader:
        imgs = imgs.to(config.DEVICE, non_blocking=True)
        logits = model(imgs).squeeze(-1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        y_probs.extend(probs.tolist())
        y_true.extend(labels.numpy().tolist())
    y_true  = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred  = (y_probs > 0.5).astype(int)
    return y_true, y_pred, y_probs


def plot_training_curves(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.get("loss", []), label="train")
    axes[0].plot(history.get("val_loss", []), label="val")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="train")
    axes[1].plot(history.get("val_accuracy", []), label="val")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()

    plt.tight_layout()
    out = os.path.join(config.PLOTS_DIR, f"{model_name}_curves.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASSES, yticklabels=config.CLASSES)
    plt.title(f"{model_name} — Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    out = os.path.join(config.PLOTS_DIR, f"{model_name}_cm.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")


def plot_roc_curve(y_true, y_probs, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} — ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(config.PLOTS_DIR, f"{model_name}_roc.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")
    return roc_auc


def get_classification_report(y_true, y_pred, model_name):
    report_str = classification_report(y_true, y_pred, target_names=config.CLASSES)
    print(f"\n=== {model_name} — Classification Report ===")
    print(report_str)

    # Save to outputs/results/{model_name}_report.txt
    out = os.path.join(config.RESULTS_DIR, f"{model_name}_report.txt")
    with open(out, "w") as f:
        f.write(f"=== {model_name} — Classification Report ===\n\n")
        f.write(report_str)
    print(f"Saved {out}")

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
    }


def evaluate_model(model, test_loader, model_name):
    """Run a model on the test set -> metrics + plots."""
    y_true, y_pred, y_probs = predict(model, test_loader)

    # Save raw predictions for later analysis (e.g. threshold tuning, ROC redraw)
    preds_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob": y_probs})
    preds_path = os.path.join(config.RESULTS_DIR, f"{model_name}_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved {preds_path}")

    metrics = get_classification_report(y_true, y_pred, model_name)
    plot_confusion_matrix(y_true, y_pred, model_name)
    metrics["auc"]      = plot_roc_curve(y_true, y_probs, model_name)
    metrics["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    return metrics


def build_comparison_table(results_dict):
    df = pd.DataFrame(results_dict).T
    cols = ["params_M", "accuracy", "precision", "recall", "f1", "auc"]
    df = df[[c for c in cols if c in df.columns]].round(4)

    print("\n=== Final Model Comparison ===")
    print(df.to_string())
    out = os.path.join(config.RESULTS_DIR, "comparison_table.csv")
    df.to_csv(out)
    print(f"\nSaved {out}")
    return df