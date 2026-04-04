#!/usr/bin/env python3
"""
Generate Hanns benchmark comparison charts for README.
Run: python3 scripts/gen_charts.py
Output: assets/benchmarks/*.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = "assets/benchmarks"
os.makedirs(OUT, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
HANNS   = "#4F8EF7"
NATIVE  = "#B0B8C8"
BG      = "#0D1117"
GRID    = "#21262D"
TEXT    = "#E6EDF3"
ACCENT  = "#3FB950"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "text.color":       TEXT,
    "grid.color":       GRID,
    "grid.linewidth":   0.8,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})


def chart_qps_comparison():
    indexes  = ["HNSW\n(ef=60)", "IVF-Flat\n(nprobe=32)", "IVF-SQ8\n(nprobe=32)"]
    hanns    = [17_814,  3_429, 11_717]
    native   = [15_918,    721,  8_278]
    speedups = [h / n for h, n in zip(hanns, native)]

    x   = np.arange(len(indexes))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(BG)

    bars_h = ax.bar(x - w/2, hanns,  w, label="Hanns",              color=HANNS,  zorder=3)
    bars_n = ax.bar(x + w/2, native, w, label="FAISS / KnowWhere C++ (8T)", color=NATIVE, zorder=3)

    ax.set_ylabel("Queries per Second (QPS)", labelpad=10)
    ax.set_title("Hanns vs FAISS / KnowWhere C++  —  SIFT-1M, x86", pad=14, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(indexes, fontsize=11)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top","right","left","bottom"]].set_visible(False)

    for bar, sp in zip(bars_h, speedups):
        h = bar.get_height()
        label = f"+{(sp-1)*100:.0f}%" if sp < 2 else f"{sp:.1f}×"
        ax.text(bar.get_x() + bar.get_width()/2, h + max(hanns)*0.015,
                label, ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=ACCENT)

    ax.legend(framealpha=0, fontsize=10)
    ax.set_ylim(0, max(hanns) * 1.18)

    for bar in list(bars_h) + list(bars_n):
        h = bar.get_height()
        if h > 500:
            ax.text(bar.get_x() + bar.get_width()/2, h * 0.5,
                    f"{h:,.0f}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    fig.tight_layout()
    path = f"{OUT}/qps_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  wrote {path}")


def chart_recall_qps():
    hanns_ef      = [16,      32,      50,     60,     100,    138   ]
    hanns_recall  = [0.8695,  0.9280,  0.9650, 0.9720, 0.9900, 0.9945]
    hanns_qps     = [16_350,  12_696,  8_754,  17_814, 4_775,  17_910]

    native_recall = [0.952]
    native_qps    = [15_918]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)

    ax.plot(hanns_recall, hanns_qps, "o-", color=HANNS, linewidth=2,
            markersize=7, label="Hanns", zorder=3)

    for ef, r, q in zip(hanns_ef, hanns_recall, hanns_qps):
        ax.annotate(f"ef={ef}", (r, q),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=TEXT, alpha=0.8)

    ax.scatter(native_recall, native_qps, marker="D", s=80,
               color=NATIVE, zorder=4, label="KnowWhere C++ (8T, ef=138)")

    ax.set_xlabel("Recall@10", labelpad=8)
    ax.set_ylabel("Queries per Second (QPS)", labelpad=8)
    ax.set_title("HNSW Recall vs QPS  —  SIFT-1M 128-dim L2, x86", pad=14,
                 fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.legend(framealpha=0, fontsize=10)

    fig.tight_layout()
    path = f"{OUT}/recall_qps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  wrote {path}")


def chart_speedup():
    labels   = ["IVF-Flat\n(batch vs 8T)", "IVF-SQ8\n(batch vs 8T)",
                "HNSW ef=60\n(batch vs 8T)", "IVF-Flat\n(serial vs 1T)"]
    speedups = [5.23, 1.42, 1.12, 6.9]
    colors   = [HANNS if s >= 2 else "#6EAAF7" for s in speedups]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG)

    bars = ax.barh(labels, speedups, color=colors, zorder=3, height=0.5)
    ax.axvline(1.0, color=NATIVE, linewidth=1.2, linestyle="--", zorder=4, label="Baseline (1×)")
    ax.set_xlabel("Speedup vs FAISS / KnowWhere C++", labelpad=8)
    ax.set_title("Hanns Speedup  —  SIFT-1M, x86", pad=14, fontsize=13, fontweight="bold")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.legend(framealpha=0, fontsize=10)

    for bar, sp in zip(bars, speedups):
        ax.text(sp + 0.05, bar.get_y() + bar.get_height()/2,
                f"{sp:.2f}×", va="center", fontsize=11,
                fontweight="bold", color=ACCENT)

    ax.set_xlim(0, max(speedups) * 1.2)
    fig.tight_layout()
    path = f"{OUT}/speedup.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  wrote {path}")


def chart_usq_quantization():
    # Cohere Wikipedia-1M, 768-dim IP, nprobe=32, x86 authority
    # IVF-PQ: 2026-04-04 authority run; IVF-USQ: 2026-04-02 authority run
    labels  = ["IVF-PQ\nm=32 (8×)", "IVF-PQ\nm=48 (5.3×)",
               "IVF-Flat\n(full prec.)", "IVF-SQ8\n(4×)",
               "USQ 4-bit\n(8×)", "USQ 8-bit\n(4×)"]
    qps     = [723,   502,   339,   605,   1_308, 1_011]
    recalls = [0.066, 0.127, 0.798, 0.805, 0.879, 0.968]
    RED     = "#F85149"
    colors  = [RED, RED, NATIVE, "#8899CC", "#6EAAF7", HANNS]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.patch.set_facecolor(BG)

    bars = ax.bar(labels, qps, color=colors, zorder=3, width=0.58)
    ax.set_ylabel("Queries per Second (QPS)", labelpad=10)
    ax.set_title("Quantization Comparison  —  Cohere Wikipedia-1M, 768-dim IP, nprobe=32, x86",
                 pad=14, fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top","right","left","bottom"]].set_visible(False)

    for bar, q, r, col in zip(bars, qps, recalls, colors):
        h = bar.get_height()
        # recall label — red for PQ (unusable), green for USQ
        r_color = RED if r < 0.2 else (ACCENT if r > 0.8 else TEXT)
        r_label = f"recall {r:.3f}" if r >= 0.2 else f"recall {r:.3f} ✗"
        ax.text(bar.get_x() + bar.get_width()/2, h + 20,
                f"{q:,}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=TEXT)
        ax.text(bar.get_x() + bar.get_width()/2, h + 75,
                r_label, ha="center", va="bottom", fontsize=8.5,
                color=r_color, fontweight="bold")

    # divider between PQ and Hanns groups
    top = max(qps) * 1.28
    ax.axvline(1.5, color=GRID, linewidth=1.2, linestyle="--", zorder=2)
    ax.axvline(3.5, color=GRID, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(0.5, top, "FAISS PQ", ha="center", fontsize=9,
            color=RED, style="italic")
    ax.text(2.5, top, "Baselines", ha="center", fontsize=9, color=NATIVE, style="italic")
    ax.text(4.5, top, "Hanns USQ", ha="center", fontsize=9,
            color=HANNS, style="italic", fontweight="bold")

    ax.set_ylim(0, max(qps) * 1.40)
    ax.tick_params(axis='x', labelsize=9.5)

    fig.tight_layout()
    path = f"{OUT}/usq_quantization.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    print("Generating Hanns benchmark charts...")
    chart_qps_comparison()
    chart_recall_qps()
    chart_speedup()
    chart_usq_quantization()
    print("Done. Files in assets/benchmarks/")
