"""
Verification Plots (Task 3)

Generates weighted 1D histograms of pT and y distributions,
overlaying the combined dataset before and after cell resampling.
The top panel shows both distributions as step histograms.
The bottom panel shows the bin-by-bin ratio (After / Before),

TO RUN: python3 plot_distributions.py [combined_before.csv] [combined_after.csv]
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt


def read_events(path):
    pt, y, weight = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            pt.append(float(row["pt"]))
            y.append(float(row["y"]))
            weight.append(float(row["weight"]))
    return np.array(pt), np.array(y), np.array(weight)


def plot_comparison(ax_main, ax_ratio, values, w_before, w_after,
                    bins, xlabel, ylabel):

    h_before, edges = np.histogram(values, bins=bins, weights=w_before)
    h_after, _      = np.histogram(values, bins=bins, weights=w_after)
    centers = 0.5 * (edges[:-1] + edges[1:])

    #the top panel, step histograms
    ax_main.stairs(h_before, edges, color="black", linewidth=1.5,
                   label="Before resampling (NLO)")
    ax_main.stairs(h_after, edges, color="#E53935", linewidth=1.5,
                   linestyle="--", label="After resampling")
    ax_main.set_ylabel(ylabel, fontsize=12)
    ax_main.legend(fontsize=11, frameon=False)
    ax_main.set_xlim(edges[0], edges[-1])
    ax_main.tick_params(labelbottom=False)

    #the bottom panel, computing ratio for bins
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(h_before) > 1.0, h_after / h_before, np.nan)

    ax_ratio.scatter(centers, ratio, s=15, color="#E53935", zorder=3)
    ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle="-")

    ax_ratio.axhspan(0.95, 1.05, color="gray", alpha=0.15)

    ax_ratio.set_ylabel("Ratio", fontsize=11)
    ax_ratio.set_xlabel(xlabel, fontsize=12)
    ax_ratio.set_ylim(0.80, 1.20)
    ax_ratio.set_xlim(edges[0], edges[-1])


#--------------main--------------
def main():
    before_path = sys.argv[1] if len(sys.argv) > 1 else "combined_before.csv"
    after_path  = sys.argv[2] if len(sys.argv) > 2 else "combined_after.csv"

    pt, y, w_before = read_events(before_path)
    _,  _, w_after  = read_events(after_path)

    n_neg_before = int(np.sum(w_before < 0))
    n_neg_after  = int(np.sum(w_after < 0))
    print(f"Loaded {len(pt)} events")
    print(f"  Before: sum(w) = {np.sum(w_before):.4f}, "
          f"{n_neg_before} negative weights")
    print(f"  After:  sum(w) = {np.sum(w_after):.4f}, "
          f"{n_neg_after} negative weights")

    #pT distribution

    pt_bins = np.linspace(0, 150, 31)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.05},
        constrained_layout=True
    )
    plot_comparison(ax1, ax2, pt, w_before, w_after,
                    pt_bins, r"$p_T$ [GeV]", r"$\sum\, w$ per bin")
    ax1.set_title(r"$p_T$ distribution: before vs after cell resampling",
                  fontsize=13, pad=10)
    fig.savefig("pt_distribution.png", dpi=200)
    print("Saved pt_distribution.png")

    #y distribution

    y_bins = np.linspace(-5, 5, 26)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.05},
        constrained_layout=True
    )
    plot_comparison(ax1, ax2, y, w_before, w_after,
                    y_bins, r"Rapidity $y$", r"$\sum\, w$ per bin")
    ax1.set_title(r"Rapidity distribution: before vs after cell resampling",
                  fontsize=13, pad=10)
    fig.savefig("y_distribution.png", dpi=200)
    print("Saved y_distribution.png")


if __name__ == "__main__":
    main()
