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
    # Cohere Wikipedia-1M, 768-dim IP, nprobe=32, x86 authority (2026-04-02)
    labels   = ["IVF-Flat\n(full precision)", "IVF-SQ8\n(8-bit, 4× compression)",
                "IVF-USQ 4-bit\n(8× compression)", "IVF-USQ 8-bit\n(4× compression)"]
    qps      = [339,  605,  1_308, 1_011]
    recalls  = [0.798, 0.805, 0.879, 0.968]
    colors   = [NATIVE, "#8899CC", "#6EAAF7", HANNS]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG)

    bars = ax.bar(labels, qps, color=colors, zorder=3, width=0.55)
    ax.set_ylabel("Queries per Second (QPS)", labelpad=10)
    ax.set_title("Quantization: QPS vs Compression  —  Cohere Wikipedia-1M, 768-dim, x86",
                 pad=14, fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top","right","left","bottom"]].set_visible(False)

    # QPS value + recall label above each bar
    for bar, q, r in zip(bars, qps, recalls):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 25,
                f"{q:,} QPS", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=TEXT)
        ax.text(bar.get_x() + bar.get_width()/2, h + 80,
                f"recall {r:.3f}", ha="center", va="bottom", fontsize=9,
                color=ACCENT)

    ax.set_ylim(0, max(qps) * 1.22)
    ax.tick_params(axis='x', labelsize=10)

    # annotation: USQ 4x vs IVF-Flat
    ax.annotate("", xy=(3 - 0.275, 1_011), xytext=(0 + 0.275, 339),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5))
    ax.text(1.5, 820, "3.0× faster\n+17% recall\n¼ the memory",
            ha="center", fontsize=9, color=ACCENT, fontweight="bold")

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
