"""Render the two manuscript figures from frozen artifacts.

Usage: python paper/figures/make_paper_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
OUT = Path(__file__).resolve().parent
THRESHOLD = 0.645

mpl.rcParams.update(
    {
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    }
)

BLUE, ORANGE, GRAY = "#1f6fb4", "#e07b39", "#8a8a8a"


def main() -> None:
    layers = pd.read_csv(ART / "qwen2.5-math-1.5b/probes/layer_metrics.csv")
    layers = layers[layers["split"] == "test"].sort_values("layer")
    traj = pd.read_csv(ART / "experiment2_cpu/error_aligned_trajectory.csv")
    transfer = pd.read_csv(ART / "qwen2.5-math-1.5b/probes/domain_transfer.csv")

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.75))

    ax = axes[0]
    ax.plot(layers["layer"], layers["auroc"], "-o", ms=2.6, color=BLUE, label="boundary AUROC")
    ax.plot(
        layers["layer"],
        layers["process_f1"],
        "-s",
        ms=2.6,
        color=ORANGE,
        label="first-error F1",
    )
    ax.axvline(23, color="k", ls=":", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("hidden-state index")
    ax.set_ylabel("held-out score")
    ax.set_title("Decoding vs. localization")
    ax.legend(loc="center left", frameon=False, handlelength=1.4)
    ax.annotate(
        "selected 23",
        xy=(23, 0.03),
        xytext=(11.0, 0.03),
        fontsize=6.5,
        color="k",
    )

    ax = axes[1]
    ax.fill_between(traj["relative_step"], traj["ci_low"], traj["ci_high"], color=BLUE, alpha=0.18)
    ax.plot(traj["relative_step"], traj["mean_score"], "-o", ms=3, color=BLUE)
    ax.axhline(THRESHOLD, color=ORANGE, ls="--", lw=1.0)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("steps from annotated first error")
    ax.set_ylabel("invalid-so-far score")
    ax.set_title("The score crosses late")
    ax.text(-3.05, THRESHOLD + 0.04, "threshold", fontsize=6.5, color=ORANGE)

    ax = axes[2]
    diag = transfer["train_source"] == transfer["test_source"]
    ax.scatter(
        transfer.loc[~diag, "auroc"],
        transfer.loc[~diag, "process_f1"],
        s=16,
        facecolors="none",
        edgecolors=BLUE,
        lw=0.9,
        label="cross-source",
    )
    ax.scatter(
        transfer.loc[diag, "auroc"],
        transfer.loc[diag, "process_f1"],
        s=18,
        color=ORANGE,
        marker="D",
        label="same source",
    )
    ax.set_xlim(0.70, 0.95)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("first-error F1")
    ax.set_title("Transfer across sources")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.2)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / "gap.pdf")
    fig.savefig(OUT / "gap.png")
    plt.close(fig)

    effects = pd.read_csv(ART / "qwen2.5-math-1.5b/interventions/effect_statistics.csv")
    effects = effects[(effects["scope"] == "overall") & (effects["statistic"] == "paired_effect")]
    fig, ax = plt.subplots(figsize=(2.9, 1.75))
    ax.errorbar(
        effects["alpha"],
        effects["estimate"],
        yerr=[
            effects["estimate"] - effects["ci_low"],
            effects["ci_high"] - effects["estimate"],
        ],
        fmt="o-",
        ms=3,
        lw=1.0,
        color=BLUE,
        capsize=2,
    )
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel(r"dose $\alpha$ (projection SD)")
    ax.set_ylabel("change in verdict score")
    ax.set_title("Intervention on an invalid readout")
    fig.tight_layout()
    fig.savefig(OUT / "intervention.pdf")
    fig.savefig(OUT / "intervention.png")
    plt.close(fig)

    print("wrote gap.pdf, intervention.pdf")


if __name__ == "__main__":
    main()
