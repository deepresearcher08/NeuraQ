"""Shared plotting helpers."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def setup_figure(figsize=(10, 4)):
    return plt.subplots(figsize=figsize)


def plot_variance_vs_n(ax, xs, ys, label=None, color="crimson", marker="o-", logy=True):
    if logy:
        ax.semilogy(xs, ys, marker, color=color, label=label)
    else:
        ax.plot(xs, ys, marker, color=color, label=label)
    ax.set_xlabel("n (qubits)")
    ax.set_ylabel("gradient variance")
    ax.grid(True, which="both", alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_training_curves(ax, steps, means, ci_low=None, ci_high=None, label=None, color="blue"):
    ax.plot(steps, means, label=label, color=color)
    if ci_low is not None and ci_high is not None:
        ax.fill_between(steps, ci_low, ci_high, alpha=0.2, color=color)
    ax.set_xlabel("step")
    ax.set_ylabel("cost")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax
