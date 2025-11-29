#!/usr/bin/env python3
"""Plot training and validation loss from CSV file."""

import argparse
import csv

import matplotlib.pyplot as plt


def plot_loss(csv_path, save_path=None):
    """从CSV文件读取loss历史并画图"""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        history = list(reader)

    if not history:
        print("No data found in CSV!")
        return

    iters = [int(h["iter"]) for h in history]
    train_loss = [float(h["train_loss"]) for h in history]
    val_loss = [float(h["val_loss"]) for h in history]
    val_ltr = [float(h["val_left_to_right"]) for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(iters, train_loss, label="Train Loss", marker="o", markersize=4)
    plt.plot(iters, val_loss, label="Val Loss (random order)", marker="s", markersize=4)
    plt.plot(iters, val_ltr, label="Val Loss (left-to-right)", marker="^", markersize=4)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss curves from CSV file")
    parser.add_argument(
        "--csv",
        default="out-shakespeare-char/loss_history.csv",
        help="Path to loss_history.csv",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save path for the figure (e.g., loss_curve.png)",
    )
    args = parser.parse_args()
    plot_loss(args.csv, args.save)
