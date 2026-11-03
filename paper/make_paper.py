# builds paper/DATS_MaxCut_paper.pdf from the CSV files in results/
# all numbers in tables, plots and text are read from the CSVs, nothing is typed by hand
#   pip install reportlab==4.0.9 matplotlib numpy
#   python3 paper/make_paper.py

import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image,
                                Table, TableStyle, KeepTogether, NextPageTemplate, FrameBreak, Preformatted)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "paper", "figures")
OUT_PDF = os.path.join(ROOT, "paper", "DATS_MaxCut_paper.pdf")
os.makedirs(FIGS, exist_ok=True)

# ---------------------------------------------------------------- fonts
# STIX fonts come with matplotlib, they look like Times and have all the math symbols
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("Stix", os.path.join(FONT_DIR, "STIXGeneral.ttf")))
pdfmetrics.registerFont(TTFont("Stix-Bold", os.path.join(FONT_DIR, "STIXGeneralBol.ttf")))
pdfmetrics.registerFont(TTFont("Stix-Italic", os.path.join(FONT_DIR, "STIXGeneralItalic.ttf")))
pdfmetrics.registerFont(TTFont("Stix-BoldItalic", os.path.join(FONT_DIR, "STIXGeneralBolIta.ttf")))
pdfmetrics.registerFont(TTFont("Mono", os.path.join(FONT_DIR, "DejaVuSansMono.ttf")))
addMapping("Stix", 0, 0, "Stix")
addMapping("Stix", 1, 0, "Stix-Bold")
addMapping("Stix", 0, 1, "Stix-Italic")
addMapping("Stix", 1, 1, "Stix-BoldItalic")

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# ---------------------------------------------------------------- data
def read_csv(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def read_text(name, default):
    path = os.path.join(RESULTS, name)
    if os.path.exists(path):
        txt = open(path).read().strip()
        if txt:
            return txt.splitlines()[0].strip()
    return default


cpu_rows = read_csv("cpu.csv")
quality_rows = read_csv("quality.csv")
gpu_rows = read_csv("gpu.csv")
gpu_thread_rows = read_csv("gpu_threads.csv")
HAVE_GPU = len(gpu_rows) > 0
CPU_NAME = read_text("cpu_name.txt", "unknown CPU")
GPU_NAME = read_text("gpu_name.txt", "NVIDIA GPU")

cpu = defaultdict(dict)  # cpu[graph][version] = row
for r in cpu_rows:
    cpu[r["graph"]][r["version"]] = r
gpu = defaultdict(dict)
for r in gpu_rows:
    gpu[r["graph"]][r["version"]] = r

GSET = ["G1", "G14", "G22", "G43", "G48", "G55"]
SNAP = ["wiki-Vote", "email-Enron", "ca-AstroPh", "soc-Epinions1"]
SYNTH = ["powerlaw-100k"]
ALL_GRAPHS = [g for g in GSET + SNAP + SYNTH if g in cpu]
SPARSE = [g for g in ALL_GRAPHS if g in ("G48", "G55") or g in SNAP or g in SYNTH]
# best known cut values for G-set as reported by Benlic and Hao [2]
BEST_KNOWN = {"G1": 11624, "G14": 3064, "G22": 13359, "G43": 6660, "G48": 6000, "G55": 10294}
GSET_TYPE = {"G1": "random", "G14": "planar", "G22": "random", "G43": "random", "G48": "toroidal", "G55": "random"}


def f(row, key):
    return float(row[key])


def speedup(graph, version, base="cpu_full"):
    return f(cpu[graph][version], "moves_per_s") / f(cpu[graph][base], "moves_per_s")


def work_ratio(graph, version="cpu_tiled"):
    return f(cpu[graph]["cpu_full"], "sel_work_per_step") / f(cpu[graph][version], "sel_work_per_step")


def deg_stats(graph):
    """degree list of a real graph read straight from the data file (for fig + table)"""
    path = os.path.join(DATA, graph + ".txt")
    if not os.path.exists(path):
        return None
    edges = set()
    with open(path) as fin:
        if graph in GSET:
            next(fin)
            for line in fin:
                p = line.split()
                if len(p) >= 2:
                    a, b = int(p[0]), int(p[1])
                    if a != b:
                        edges.add((min(a, b), max(a, b)))
        else:
            for line in fin:
                if line[0] in "#%":
                    continue
                p = line.split()
                if len(p) >= 2:
                    a, b = int(p[0]), int(p[1])
                    if a != b:
                        edges.add((min(a, b), max(a, b)))
    deg = defaultdict(int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return np.array(list(deg.values()))


DEGREES = {g: deg_stats(g) for g in ALL_GRAPHS}

# ---------------------------------------------------------------- numbers used in the text
sparse_work = [work_ratio(g) for g in SPARSE]
sparse_tiled_speed = [speedup(g, "cpu_tiled") for g in SPARSE]
omp_full_speed = [speedup(g, "cpu_full_omp") for g in SPARSE]
omp_dats_speed = [speedup(g, "cpu_tiled_bfs_omp") for g in SPARSE]
dense = [g for g in ("G1", "G43") if g in cpu]
num_threads_cpu = 10  # cores of the laptop used, also number of OpenMP threads
shuffled = [g for g in SNAP if g + "-shuffled" in cpu]
reorder_gain = []
for g in shuffled:
    s = cpu[g + "-shuffled"]
    reorder_gain.append(1 - f(s["cpu_tiled_bfs"], "dirty_tiles_per_step") / f(s["cpu_tiled"], "dirty_tiles_per_step"))
gset_reorder = [100 * abs(1 - f(cpu[g]["cpu_tiled_bfs"], "dirty_tiles_per_step") / f(cpu[g]["cpu_tiled"], "dirty_tiles_per_step"))
                for g in GSET if g in cpu]
# only graphs that really have hubs (degree > 1024)
hub_pct = [f(cpu[g]["cpu_full"], "hub_flip_pct") for g in SNAP + SYNTH if g in cpu and int(cpu[g]["cpu_full"]["maxdeg"]) > 1024]
STEPS = int(cpu_rows[0]["steps"]) if cpu_rows else 0
SEARCHES = int(cpu_rows[0]["searches"]) if cpu_rows else 0

quality = defaultdict(dict)
for r in quality_rows:
    quality[r["graph"]][r["version"]] = r
Q_STEPS = int(quality_rows[0]["steps"]) if quality_rows else 0
Q_SEARCHES = int(quality_rows[0]["searches"]) if quality_rows else 0


def rng_txt(vals, fmt="{:.1f}"):
    return (fmt + "–" + fmt).format(min(vals), max(vals))


# GPU numbers (only if gpu.csv is there)
GPU_VERS = ["gpu_naive_full", "gpu_degree_full", "gpu_degree_tiled", "gpu_naive_tiled_bfs", "gpu_dats"]
GPU_LABEL = {"gpu_naive_full": "Naive", "gpu_degree_full": "DA", "gpu_degree_tiled": "DA+T",
             "gpu_naive_tiled_bfs": "T+R", "gpu_dats": "DATS"}
gpu_graphs = [g for g in ALL_GRAPHS if g in gpu and all(v in gpu[g] for v in GPU_VERS)]
# on the GPU the result depends a lot on graph size, so we talk about large and small graphs separately
LARGE = [g for g in gpu_graphs if int(gpu[g]["gpu_dats"]["n"]) >= 15000]
SMALL = [g for g in gpu_graphs if int(gpu[g]["gpu_dats"]["n"]) <= 2000]


def gspeed(graph, version, base="gpu_naive_full"):
    return f(gpu[graph][version], "moves_per_s") / f(gpu[graph][base], "moves_per_s")


# ---------------------------------------------------------------- figures
COL_W_IN = 3.45
BLUE, ORANGE, GREEN, RED, GREY = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096"


def save(fig, name):
    path = os.path.join(FIGS, name)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_timeline():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W_IN, 1.45), sharey=True)
    work = [3, 2, 40, 4]
    for ax, title, balanced in ((axes[0], "(a) one thread does all neighbours", False),
                                (axes[1], "(b) degree aware sharing", True)):
        total = sum(work)
        end = total / 4 if balanced else max(work)
        for t in range(4):
            busy = total / 4 if balanced else work[t]
            ax.barh(t, busy, color=BLUE if not (t == 2 and not balanced) else RED, height=0.55)
            if busy < end:
                ax.barh(t, end - busy, left=busy, color="white", edgecolor=GREY, hatch="////", height=0.55, lw=0.4)
        ax.axvline(end, color="k", lw=0.7, ls="--")
        ax.set_xlim(0, 44)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("time (neighbour updates)")
        ax.set_yticks(range(4))
        ax.set_yticklabels([f"thread {i + 1}" for i in range(4)])
        ax.invert_yaxis()
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].text(21, 2, "hub vertex", color="white", va="center", ha="center", fontsize=7)
    fig.tight_layout(w_pad=0.6)
    return save(fig, "fig1_timeline.png")


def box(ax, x, y, w, h, text, fc="#ebf4ff", ec=BLUE, fs=6.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="k"))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(COL_W_IN, 2.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # host
    box(ax, 0.02, 0.83, 0.96, 0.16, "", fc="#f7fafc", ec=GREY)
    ax.text(0.04, 0.975, "Host (CPU)", fontsize=7, weight="bold", va="top")
    steps = ["read graph\n+ build CSR", "BFS\nrelabel", "degree\nclasses", "collect best\n+ verify cut"]
    for i, s in enumerate(steps):
        box(ax, 0.06 + i * 0.235, 0.84, 0.19, 0.085, s, fs=6.2)
    for i in range(2):
        arrow(ax, 0.25 + i * 0.235, 0.882, 0.295 + i * 0.235, 0.882)
    # global memory
    box(ax, 0.02, 0.66, 0.96, 0.12, "GPU global memory: row_ptr, col_idx, weights, degree class (read by all blocks)\n"
                                    "per block scratch: tabu list (+ gains if they do not fit in shared memory)",
        fc="#fffaf0", ec=ORANGE, fs=5.9)
    arrow(ax, 0.25, 0.83, 0.25, 0.785)
    # blocks
    for i, x in enumerate((0.02, 0.35, 0.68)):
        if i == 2:
            box(ax, x, 0.04, 0.30, 0.56, "", fc="#f7fafc", ec=GREY)
            ax.text(x + 0.15, 0.32, "... one block\nper independent\nsearch", ha="center", va="center", fontsize=7)
            continue
        box(ax, x, 0.04, 0.30, 0.56, "", fc="#f0fff4", ec=GREEN)
        ax.text(x + 0.15, 0.565, f"Thread block {i + 1}", ha="center", fontsize=7, weight="bold")
        items = ["tile summaries\n(best free / tabu key)", "dirty tile bitmask", "solution, 1 bit/vertex", "gains (if they fit)"]
        for k, it in enumerate(items):
            box(ax, x + 0.02, 0.40 - k * 0.085, 0.26, 0.07, it, fc="white", ec=GREEN, fs=5.8)
        ax.text(x + 0.15, 0.09, "shared memory", ha="center", fontsize=6, style="italic", color=GREEN)
        arrow(ax, x + 0.15, 0.66, x + 0.15, 0.60)
    arrow(ax, 0.82, 0.60, 0.82, 0.66)
    fig.tight_layout()
    return save(fig, "fig2_architecture.png")


def fig_tiles():
    fig, ax = plt.subplots(figsize=(COL_W_IN, 2.15))
    ax.set_xlim(-0.3, 8.3)
    ax.set_ylim(-2.75, 2.5)
    ax.axis("off")
    rng = np.random.RandomState(3)
    gains = rng.randint(-4, 6, size=(8, 4))
    dirty = {1, 4, 6}
    for t in range(8):
        for k in range(4):
            fc = "#fed7d7" if (t, k) == (4, 1) else ("#fefcbf" if t in dirty else "white")
            ax.add_patch(Rectangle((t + k * 0.22, 1.2), 0.22, 0.5, fc=fc, ec="k", lw=0.4))
            ax.text(t + k * 0.22 + 0.11, 1.45, str(gains[t, k]), ha="center", va="center", fontsize=5.5)
        ax.text(t + 0.44, 1.85, f"tile {t}", ha="center", fontsize=6)
        summary_fc = "#fefcbf" if t in dirty else "#ebf4ff"
        ax.add_patch(Rectangle((t + 0.05, 0.0), 0.78, 0.45, fc=summary_fc, ec=BLUE, lw=0.6))
        ax.text(t + 0.44, 0.225, f"max {gains[t].max()}", ha="center", va="center", fontsize=5.6)
        ax.annotate("", xy=(t + 0.44, 0.47), xytext=(t + 0.44, 1.18),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=RED if t in dirty else GREY))
    ax.text(4.0 + 0.33, 2.2, "flipped vertex v", ha="center", fontsize=6.5, color=RED)
    ax.annotate("", xy=(4 + 0.33, 1.72), xytext=(4.33, 2.12), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=RED))
    ax.text(-0.25, 1.45, "gains", ha="right", va="center", fontsize=6.5)
    ax.text(-0.25, 0.225, "tile\nsummary", ha="right", va="center", fontsize=6.5)
    ax.add_patch(Rectangle((6.3, 2.12), 0.2, 0.18, fc="#fefcbf", ec="k", lw=0.4))
    ax.text(6.55, 2.21, "dirty tile (one warp each)", va="center", fontsize=5.8)
    ax.add_patch(FancyBboxPatch((1.2, -1.35), 6.0, 0.55, boxstyle="round,pad=0.02", fc="#f0fff4", ec=GREEN, lw=0.8))
    ax.text(4.2, -1.075, "block argmax over n/32 summaries (warp shuffle)", ha="center", va="center", fontsize=6.3)
    for t in range(8):
        ax.annotate("", xy=(4.2 + (t - 3.5) * 0.3, -0.6), xytext=(t + 0.44, -0.02),
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=GREY))
    ax.annotate("", xy=(4.2, -2.0), xytext=(4.2, -1.38), arrowprops=dict(arrowstyle="-|>", lw=0.8))
    ax.text(4.2, -2.3, "next move", ha="center", fontsize=6.8, weight="bold")
    return save(fig, "fig3_tiles.png")


def fig_degree():
    fig, ax = plt.subplots(figsize=(COL_W_IN, 1.9))
    styles = [(BLUE, "-"), (ORANGE, "-"), (GREEN, "-"), (RED, "-"), (GREY, "--")]
    for (c, ls), g in zip(styles, SNAP + ["G55"]):
        d = DEGREES.get(g)
        if d is None:
            continue
        d = np.sort(d)
        ccdf = 1.0 - np.arange(len(d)) / len(d)
        ax.loglog(d, ccdf, color=c, ls=ls, lw=1.0, label=g)
    ax.axvline(32, color="k", lw=0.5, ls=":")
    ax.axvline(1024, color="k", lw=0.5, ls=":")
    ax.text(33, 0.5, "32", fontsize=6)
    ax.text(1050, 0.5, "1024", fontsize=6)
    ax.set_xlabel("degree d")
    ax.set_ylabel("P(degree ≥ d)")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    return save(fig, "fig4_degree.png")


def fig_selection_work():
    fig, ax = plt.subplots(figsize=(COL_W_IN, 1.9))
    graphs = ALL_GRAPHS
    x = np.arange(len(graphs))
    full = [f(cpu[g]["cpu_full"], "sel_work_per_step") for g in graphs]
    tiled = [f(cpu[g]["cpu_tiled"], "sel_work_per_step") for g in graphs]
    ax.bar(x - 0.2, full, 0.4, color=GREY, label="full scan")
    ax.bar(x + 0.2, tiled, 0.4, color=BLUE, label="tiled selection")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("soc-", "").replace("powerlaw-", "PL-") for g in graphs], rotation=40, ha="right")
    ax.set_ylabel("entries read per step")
    for xi, a, b in zip(x, full, tiled):
        ax.text(xi + 0.2, b * 1.15, f"{a / b:.0f}×", ha="center", fontsize=5.5)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig5_selection_work.png")


def fig_cpu_speedup():
    fig, ax = plt.subplots(figsize=(COL_W_IN, 1.9))
    graphs = ALL_GRAPHS
    x = np.arange(len(graphs))
    vers = [("cpu_tiled", "tiled (1 thread)", BLUE), ("cpu_full_omp", f"full scan, OpenMP", GREY),
            ("cpu_tiled_bfs_omp", "tiled + BFS, OpenMP", GREEN)]
    for i, (v, lab, c) in enumerate(vers):
        ax.bar(x + (i - 1) * 0.27, [speedup(g, v) for g in graphs], 0.27, color=c, label=lab)
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("soc-", "").replace("powerlaw-", "PL-") for g in graphs], rotation=40, ha="right")
    ax.set_ylabel("speedup over serial full scan")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig6_cpu_speedup.png")


def fig_reorder():
    fig, ax = plt.subplots(figsize=(COL_W_IN, 1.6))
    x = np.arange(len(shuffled))
    crawl = [f(cpu[g]["cpu_tiled"], "dirty_tiles_per_step") for g in shuffled]
    shuf = [f(cpu[g + "-shuffled"]["cpu_tiled"], "dirty_tiles_per_step") for g in shuffled]
    bfs = [f(cpu[g + "-shuffled"]["cpu_tiled_bfs"], "dirty_tiles_per_step") for g in shuffled]
    ax.bar(x - 0.27, shuf, 0.27, color=RED, label="random ids")
    ax.bar(x, crawl, 0.27, color=GREY, label="ids from SNAP file")
    ax.bar(x + 0.27, bfs, 0.27, color=GREEN, label="random ids + BFS relabel")
    ax.set_xticks(x)
    ax.set_xticklabels(shuffled)
    ax.set_ylabel("dirty tiles per step")
    ax.legend(frameon=False, fontsize=6.3, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig7_reorder.png")


def fig_gpu():
    fig, ax = plt.subplots(figsize=(COL_W_IN, 1.95))
    x = np.arange(len(gpu_graphs))
    cols = [GREY, ORANGE, BLUE, "#805ad5", GREEN]
    for i, (v, c) in enumerate(zip(GPU_VERS, cols)):
        ax.bar(x + (i - 2) * 0.16, [gspeed(g, v) for g in gpu_graphs], 0.16, color=c, label=GPU_LABEL[v])
    ax.axhline(1, color="k", lw=0.5)
    ax.set_yscale("log")
    ax.set_ylim(0.2, 40)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("soc-", "").replace("powerlaw-", "PL-") for g in gpu_graphs], rotation=40, ha="right")
    ax.set_ylabel("speedup vs naive GPU")
    ax.legend(frameon=False, ncol=5, loc="upper left", fontsize=6)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig8_gpu.png")


def fig_gpu_threads():
    rows = [r for r in gpu_thread_rows if r["version"] in ("gpu_degree_full", "gpu_dats")]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(COL_W_IN, 1.6))
    for v, c in (("gpu_degree_full", ORANGE), ("gpu_dats", GREEN)):
        pts = sorted((int(r["graph"].split("tpb")[1]), f(r, "moves_per_s")) for r in rows if r["version"] == v)
        ax.plot([p[0] for p in pts], [p[1] / 1e6 for p in pts], marker="o", ms=3, color=c, label=GPU_LABEL[v])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("threads per block")
    ax.set_ylabel("million moves / s")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    return save(fig, "fig9_gpu_threads.png")


print("making figures...")
FIG = {
    "timeline": fig_timeline(), "arch": fig_architecture(), "tiles": fig_tiles(), "degree": fig_degree(),
    "work": fig_selection_work(), "cpu": fig_cpu_speedup(), "reorder": fig_reorder(),
}
if gpu_graphs:
    FIG["gpu"] = fig_gpu()
    FIG["gpu_threads"] = fig_gpu_threads()

# ---------------------------------------------------------------- extra experiments (run_experiments.sh extra)
threads_rows = read_csv("cpu_threads.csv")
scaling_rows = read_csv("cpu_scaling.csv")
tilesize_rows = read_csv("cpu_tilesize.csv")
conv_rows = read_csv("convergence.csv")


def by_graph(rows):
    out = defaultdict(dict)
    for r in rows:
        out[r["graph"]][r["version"]] = r
    return out


THR = by_graph(threads_rows)
SCALE = by_graph(scaling_rows)
TILE = by_graph(tilesize_rows)
CONV = by_graph(conv_rows)
THREAD_COUNTS = sorted({int(g.split("-omp")[1]) for g in THR})
SCALE_N = sorted({int(g.split("-")[1]) for g in SCALE})
TILE_SIZES = sorted({int(g.split("-T")[1]) for g in TILE})
TILE_GRAPHS = [g for g in ["wiki-Vote", "email-Enron", "soc-Epinions1", "G55"] if f"{g}-T32" in TILE]
CONV_STEPS = sorted({int(g.split("-s")[-1]) for g in CONV})
T4_SHARED = 64 * 1024 - 2048  # what our code can use per block on a T4 (opt-in limit minus static part)


def shared_layout(n):
    tiles = (n + 31) // 32
    parts = {"tile summaries": 20 * tiles, "dirty bitmask": 4 * ((tiles + 31) // 32), "solution bits": 4 * ((n + 31) // 32)}
    used = sum(parts.values())
    gains_fit = used + 4 * n <= T4_SHARED
    if gains_fit:
        parts["gains"] = 4 * n
    return parts, gains_fit


def fig_gpu_basics():
    fig, ax = plt.subplots(figsize=(COL_W_IN, 2.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.01, 0.2, 0.33, 0.78, "", fc="#f7fafc", ec=GREY)
    ax.text(0.175, 0.93, "Grid (one kernel launch)", ha="center", fontsize=6.5, weight="bold")
    for k in range(6):
        r, c = divmod(k, 2)
        box(ax, 0.04 + c * 0.15, 0.66 - r * 0.2, 0.12, 0.15, f"block {k}", fc="#f0fff4", ec=GREEN, fs=5.8)
    box(ax, 0.42, 0.2, 0.57, 0.78, "", fc="#f0fff4", ec=GREEN)
    ax.text(0.705, 0.93, "One thread block (runs on one SM)", ha="center", fontsize=6.5, weight="bold")
    for w in range(3):
        y = 0.75 - w * 0.13
        ax.text(0.44, y + 0.04, f"warp {w}" if w < 2 else "...", fontsize=5.8, va="center")
        if w < 2:
            for t in range(16):
                ax.add_patch(Rectangle((0.53 + t * 0.026, y), 0.022, 0.08, fc="#ebf4ff", ec=BLUE, lw=0.4))
            ax.text(0.53 + 16 * 0.026 + 0.005, y + 0.04, "32 lanes", fontsize=5.2, va="center")
    box(ax, 0.45, 0.25, 0.51, 0.16, "shared memory (64 KB on T4, fast)", fc="white", ec=GREEN, fs=5.8)
    ax.annotate("", xy=(0.17, 0.2), xytext=(0.17, 0.12), arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax.annotate("", xy=(0.7, 0.2), xytext=(0.7, 0.12), arrowprops=dict(arrowstyle="<->", lw=0.7))
    box(ax, 0.01, 0.0, 0.98, 0.11, "global memory (16 GB on T4, slow, shared by all blocks)", fc="#fffaf0", ec=ORANGE, fs=6.2)
    ax.annotate("shuffle", xy=(0.6, 0.62), xytext=(0.6, 0.48), fontsize=5.5, ha="center",
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=RED), color=RED)
    return save(fig, "fig_gpu_basics.png")


# small example graph for the worked example (gains are computed here, not typed)
EX_POS = {"A": (0, 1), "B": (1, 1.6), "C": (1, 0.4), "D": (2.2, 1.6), "E": (2.2, 0.4), "F": (3.2, 1.0)}
EX_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("B", "E"), ("C", "E"), ("D", "E"), ("D", "F"), ("E", "F")]
EX_SIDE0 = {"A": 0, "B": 0, "C": 1, "D": 0, "E": 0, "F": 1}


def ex_gains(side):
    g = {v: 0 for v in EX_POS}
    for a, b in EX_EDGES:
        d = 1 if side[a] == side[b] else -1
        g[a] += d
        g[b] += d
    return g


def ex_cut(side):
    return sum(1 for a, b in EX_EDGES if side[a] != side[b])


EX_G0 = ex_gains(EX_SIDE0)
EX_FLIP = max(sorted(EX_POS), key=lambda v: EX_G0[v])
EX_SIDE1 = dict(EX_SIDE0)
EX_SIDE1[EX_FLIP] ^= 1
EX_G1 = ex_gains(EX_SIDE1)
EX_NEIGH = sorted({b for a, b in EX_EDGES if a == EX_FLIP} | {a for a, b in EX_EDGES if b == EX_FLIP})


def fig_example():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W_IN, 1.55))
    for ax, side, gains, title in ((axes[0], EX_SIDE0, EX_G0, f"(a) before, cut = {ex_cut(EX_SIDE0)}"),
                                   (axes[1], EX_SIDE1, EX_G1, f"(b) after flipping {EX_FLIP}, cut = {ex_cut(EX_SIDE1)}")):
        for a, b in EX_EDGES:
            (x1, y1), (x2, y2) = EX_POS[a], EX_POS[b]
            cut = side[a] != side[b]
            ax.plot([x1, x2], [y1, y2], color="k" if cut else GREY, lw=1.1 if cut else 0.7, ls="-" if cut else "--", zorder=1)
        for v, (x, y) in EX_POS.items():
            changed = (ax is axes[1]) and (v == EX_FLIP or v in EX_NEIGH)
            ax.scatter([x], [y], s=150, c="k" if side[v] else "white", edgecolors=RED if changed else "k",
                       linewidths=1.6 if changed else 0.8, zorder=2)
            ax.text(x, y, v, ha="center", va="center", fontsize=6, color="white" if side[v] else "k", zorder=3)
            ax.text(x, y - 0.33, f"g={gains[v]:+d}", ha="center", fontsize=5.6, color=RED if changed else "k")
        ax.set_title(title, fontsize=7)
        ax.set_xlim(-0.4, 3.6)
        ax.set_ylim(-0.1, 1.95)
        ax.axis("off")
    fig.tight_layout(w_pad=0.3)
    return save(fig, "fig_example.png")


def fig_memlayout():
    graphs = [g for g in ["G55", "wiki-Vote", "ca-AstroPh", "email-Enron", "soc-Epinions1", "powerlaw-100k"] if g in cpu]
    fig, ax = plt.subplots(figsize=(COL_W_IN, 1.75))
    colors_ = {"tile summaries": BLUE, "dirty bitmask": ORANGE, "solution bits": GREEN, "gains": GREY}
    for i, g in enumerate(graphs):
        parts, fit = shared_layout(int(cpu[g]["cpu_full"]["n"]))
        left = 0
        for name, val in parts.items():
            ax.barh(i, val / 1024, left=left / 1024, color=colors_[name], height=0.6,
                    label=name if i == 0 or (name == "gains" and fit and "gains" not in ax.get_legend_handles_labels()[1]) else None)
            left += val
        ax.text(left / 1024 + 0.8, i, "gains in shared" if fit else "gains in global", va="center", fontsize=5.6)
    ax.axvline(T4_SHARED / 1024, color=RED, lw=0.8, ls="--")
    ax.text(T4_SHARED / 1024 - 0.5, len(graphs) - 0.45, "T4 limit", color=RED, fontsize=6, ha="right")
    ax.set_yticks(range(len(graphs)))
    ax.set_yticklabels(graphs)
    ax.set_xlabel("shared memory per block (KB)")
    ax.set_xlim(0, 80)
    ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=5.8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig_memlayout.png")


def work_parts(g):
    r = cpu[g]["cpu_tiled"]
    top = (int(r["n"]) + 31) // 32
    dirty = 32 * f(r, "dirty_tiles_per_step")
    other = max(0.0, f(r, "sel_work_per_step") - top - dirty)
    return top, dirty, other


def fig_breakdown():
    graphs = SPARSE
    fig, ax = plt.subplots(figsize=(COL_W_IN, 1.75))
    x = np.arange(len(graphs))
    bottoms = np.zeros(len(graphs))
    for idx, (lab, c) in enumerate((("scan of tile summaries", BLUE), ("dirty tile refresh", ORANGE),
                                    ("expired tiles + rebuilds", GREY))):
        vals = []
        for g in graphs:
            parts = work_parts(g)
            vals.append(100 * parts[idx] / sum(parts))
        ax.bar(x, vals, 0.6, bottom=bottoms, color=c, label=lab)
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("soc-", "").replace("powerlaw-", "PL-") for g in graphs], rotation=35, ha="right")
    ax.set_ylabel("% of selection work")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig_breakdown.png")


def fig_scaling():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W_IN, 1.6))
    ns = [n for n in SCALE_N if all(v in SCALE[f"pl-{n}"] for v in ("cpu_full", "cpu_tiled", "cpu_tiled_bfs"))]
    for v, lab, c in (("cpu_full", "full scan", GREY), ("cpu_tiled", "tiled", BLUE), ("cpu_tiled_bfs", "tiled + BFS", GREEN)):
        us = [1e6 * f(SCALE[f"pl-{n}"][v], "time_s") / f(SCALE[f"pl-{n}"][v], "moves") for n in ns]
        axes[0].loglog(ns, us, marker="o", ms=2.5, lw=1, color=c, label=lab)
        if v != "cpu_full":
            axes[1].semilogx(ns, [f(SCALE[f"pl-{n}"][v], "dirty_tiles_per_step") for n in ns], marker="o", ms=2.5,
                             lw=1, color=c, label=lab)
    axes[0].set_xlabel("vertices n")
    axes[0].set_ylabel("μs per move")
    axes[0].set_title("(a) time per move", fontsize=7)
    axes[1].set_xlabel("vertices n")
    axes[1].set_ylabel("dirty tiles per step")
    axes[1].set_title("(b) dirty tiles", fontsize=7)
    axes[0].legend(frameon=False, fontsize=5.6)
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(w_pad=0.8)
    return save(fig, "fig_scaling.png")


def fig_tilesize():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W_IN, 1.6))
    cols = [BLUE, ORANGE, GREEN, RED]
    for g, c in zip(TILE_GRAPHS, cols):
        ts = [t for t in TILE_SIZES if f"{g}-T{t}" in TILE]
        work = [f(TILE[f"{g}-T{t}"]["cpu_tiled"], "sel_work_per_step") for t in ts]
        times = [f(TILE[f"{g}-T{t}"]["cpu_tiled"], "time_s") for t in ts]
        base = f(TILE[f"{g}-T32"]["cpu_tiled"], "time_s")
        axes[0].semilogx(ts, work, base=2, marker="o", ms=2.5, lw=1, color=c, label=g)
        axes[1].semilogx(ts, [t / base for t in times], base=2, marker="o", ms=2.5, lw=1, color=c, label=g)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("tile size T")
    axes[0].set_ylabel("entries read per step")
    axes[0].set_title("(a) selection work", fontsize=7)
    axes[1].set_xlabel("tile size T")
    axes[1].set_ylabel("time / time at T=32")
    axes[1].set_title("(b) run time", fontsize=7)
    axes[1].axhline(1, color="k", lw=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=5.8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(w_pad=0.8)
    return save(fig, "fig_tilesize.png")


def omp_speedup(graph, version, t):
    return f(THR[f"{graph}-omp{t}"][version], "moves_per_s") / f(THR[f"{graph}-omp1"][version], "moves_per_s")


def fig_omp():
    fig, ax = plt.subplots(figsize=(COL_W_IN, 1.7))
    styles = {("email-Enron", "cpu_full_omp"): (GREY, "--"), ("email-Enron", "cpu_tiled_bfs_omp"): (GREEN, "--"),
              ("soc-Epinions1", "cpu_full_omp"): (GREY, "-"), ("soc-Epinions1", "cpu_tiled_bfs_omp"): (GREEN, "-")}
    for (g, v), (c, ls) in styles.items():
        ax.plot(THREAD_COUNTS, [omp_speedup(g, v, t) for t in THREAD_COUNTS], color=c, ls=ls, marker="o", ms=2.5,
                lw=1, label=f"{g}, {'full scan' if v == 'cpu_full_omp' else 'tiled + BFS'}")
    ax.plot(THREAD_COUNTS, THREAD_COUNTS, color="k", lw=0.5, ls=":", label="ideal")
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("speedup over 1 thread")
    ax.legend(frameon=False, fontsize=5.6)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    return save(fig, "fig_omp.png")


def fig_convergence():
    graphs = [g for g in ["G55", "email-Enron"] if f"{g}-s{CONV_STEPS[0]}" in CONV]
    fig, axes = plt.subplots(1, len(graphs), figsize=(COL_W_IN, 1.6))
    for ax, g in zip(np.atleast_1d(axes), graphs):
        for v, lab, c in (("cpu_full_omp", "full scan", GREY), ("cpu_tiled_bfs_omp", "tiled + BFS", GREEN)):
            pts = [(f(CONV[f"{g}-s{s}"][v], "time_s"), int(CONV[f"{g}-s{s}"][v]["best_cut"])) for s in CONV_STEPS]
            ax.semilogx([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=2.5, lw=1, color=c, label=lab)
        if g in BEST_KNOWN:
            ax.axhline(BEST_KNOWN[g], color=RED, lw=0.6, ls=":")
            ax.text(ax.get_xlim()[0] * 1.2, BEST_KNOWN[g], "best known", color=RED, fontsize=5.5, va="bottom")
        ax.set_title(g, fontsize=7)
        ax.set_xlabel("wall time (s)")
        ax.grid(alpha=0.25, lw=0.4)
    np.atleast_1d(axes)[0].set_ylabel("best cut")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=5.6, loc="lower right")
    fig.tight_layout(w_pad=0.6)
    return save(fig, "fig_convergence.png")


FIG["gpu_basics"] = fig_gpu_basics()
FIG["example"] = fig_example()
FIG["memlayout"] = fig_memlayout()
FIG["breakdown"] = fig_breakdown()
if SCALE:
    FIG["scaling"] = fig_scaling()
if TILE:
    FIG["tilesize"] = fig_tilesize()
if THR:
    FIG["omp"] = fig_omp()
if CONV:
    FIG["convergence"] = fig_convergence()


def make_equation(latex, name):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=9.5)
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    return path


EQ = {
    "cut": make_equation(r"\mathrm{cut}(x)=\sum_{(i,j)\in E} w_{ij}\,[x_i\neq x_j]", "eq1.png"),
    "gain": make_equation(r"g_v=\sum_{u\in N(v)} w_{uv}\,(1-2\,[x_u\neq x_v])", "eq2.png"),
    "update": make_equation(r"g_u \leftarrow g_u-2\,w_{uv}\,\sigma_{uv},\quad \sigma_{uv}=+1 \ \mathrm{if}\ x_u=x_v \ \mathrm{else}\ -1", "eq3.png"),
    "key": make_equation(r"K(v)=\left((g_v \oplus 2^{31})\ll 32\right)\ \vert\ \left((v\cdot\phi\ \mathrm{mod}\ 2^{32})\oplus s\right)", "eq4.png"),
    "cost": make_equation(r"W_{\mathrm{step}}=\lceil n/32\rceil + 32\,D(v) + 32\,X_t", "eq5.png"),
    "tile": make_equation(r"W(T)\approx \lceil n/T\rceil + T\cdot D_T(v)", "eq6.png"),
}

# ---------------------------------------------------------------- styles
BODY = ParagraphStyle("body", fontName="Stix", fontSize=9.6, leading=11.4, alignment=TA_JUSTIFY, firstLineIndent=10)
BODY_NI = ParagraphStyle("body_ni", parent=BODY, firstLineIndent=0)
BULLET = ParagraphStyle("bullet", parent=BODY, firstLineIndent=0, leftIndent=12, bulletIndent=3, spaceAfter=1)
H1 = ParagraphStyle("h1", fontName="Stix", fontSize=9.6, leading=12, alignment=TA_CENTER, spaceBefore=7, spaceAfter=3, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="Stix-Italic", fontSize=9.6, leading=11.4, spaceBefore=4, spaceAfter=2, keepWithNext=1)
TITLE = ParagraphStyle("title", fontName="Stix", fontSize=21, leading=25, alignment=TA_CENTER, spaceAfter=10)
AUTHOR = ParagraphStyle("author", fontName="Stix", fontSize=10.5, leading=13, alignment=TA_CENTER)
AFFIL = ParagraphStyle("affil", fontName="Stix-Italic", fontSize=9, leading=11, alignment=TA_CENTER)
ABSTRACT = ParagraphStyle("abs", fontName="Stix-Bold", fontSize=8.8, leading=10.4, alignment=TA_JUSTIFY, firstLineIndent=10)
CAPTION = ParagraphStyle("cap", fontName="Stix", fontSize=7.8, leading=9.2, alignment=TA_JUSTIFY, spaceBefore=2, spaceAfter=7)
TCAPTION = ParagraphStyle("tcap", fontName="Stix", fontSize=7.8, leading=9.4, alignment=TA_CENTER, spaceBefore=5, spaceAfter=3)
REF = ParagraphStyle("ref", fontName="Stix", fontSize=7.8, leading=9.2, alignment=TA_LEFT, leftIndent=14, firstLineIndent=-14, spaceAfter=1.5)
CODE = ParagraphStyle("code", fontName="Mono", fontSize=6.3, leading=7.6)
PENDING = ParagraphStyle("pending", parent=BODY_NI, textColor=colors.HexColor("#c53030"), fontName="Stix-Bold")

story = []
fig_no = [0]
tab_no = [0]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
sec_no = [0]
sub_no = [0]


def resolve(text):
    # [[fig:name]] and [[tab:name]] are replaced by the real figure / table number
    text = re.sub(r"\[\[fig:(\w+)\]\]", lambda m: str(FN(m.group(1))), text)
    return re.sub(r"\[\[tab:(\w+)\]\]", lambda m: TN(m.group(1)), text)


def P(text, style=BODY):
    story.append(Paragraph(resolve(text), style))


def section(title):
    sub_no[0] = 0
    story.append(Paragraph(f"{ROMAN[sec_no[0]]}. {small_caps(title)}", H1))
    sec_no[0] += 1


def small_caps(title):
    out = []
    for word in title.split():
        out.append(word[0].upper() + f'<font size="7.6">{word[1:].upper()}</font>')
    return " ".join(out)


def subsection(title):
    story.append(Paragraph(f"{chr(65 + sub_no[0])}. {title}", H2))
    sub_no[0] += 1


# order in which figures and tables appear, so the text can refer to them by name (FN / TN)
# before they are placed. figure() and table() check that the order really matches.
FIG_ORDER = [k for k in ["timeline", "gpu_basics", "example", "arch", "tiles", "memlayout", "degree", "work",
                         "breakdown", "cpu", "omp", "scaling", "tilesize", "reorder", "gpu", "gpu_threads",
                         "convergence"] if FIG.get(k)]
TAB_ORDER = ["notation", "datasets", "cost"] + (["gpu", "gpu_reorder", "cpu_gpu"] if gpu_graphs else ["gpu"]) + \
            ["quality", "params"]


def FN(key):
    return FIG_ORDER.index(key) + 1


def TN(key):
    return ROMAN[TAB_ORDER.index(key)]


def figure(key, caption, width=COL_W_IN * inch):
    path = FIG[key]
    fig_no[0] += 1
    assert fig_no[0] == FN(key), f"figure {key} is placed as {fig_no[0]} but FIG_ORDER says {FN(key)}"
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    img = Image(path, width=width, height=width * ih / iw)
    story.append(KeepTogether([Spacer(1, 3), img, Paragraph(f"Fig. {fig_no[0]}. {resolve(caption)}", CAPTION)]))
    return fig_no[0]


def equation(path, number):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    h = ih / 400 * 72 * 0.95
    w = iw / 400 * 72 * 0.95
    t = Table([[Image(path, width=w, height=h), Paragraph(f"({number})", BODY_NI)]],
              colWidths=[COL_W_IN * inch - 22, 22])
    t.setStyle(TableStyle([("ALIGN", (0, 0), (0, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story.append(t)


def table(caption, data, col_widths, font_size=7.0, bold_last_col=False, highlight_rows=(), key=None):
    tab_no[0] += 1
    if key:
        assert ROMAN[tab_no[0] - 1] == TN(key), f"table {key} is placed as {tab_no[0]} but TAB_ORDER says {TN(key)}"
    cell = ParagraphStyle("cell", fontName="Stix", fontSize=font_size, leading=font_size + 1.4, alignment=TA_CENTER)
    head = ParagraphStyle("head", parent=cell, fontName="Stix-Bold")
    rows = []
    for i, r in enumerate(data):
        rows.append([Paragraph(str(c), head if i == 0 else cell) for c in r])
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
             ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
             ("LEFTPADDING", (0, 0), (-1, -1), 1.5), ("RIGHTPADDING", (0, 0), (-1, -1), 1.5)]
    for r in highlight_rows:
        style.append(("LINEABOVE", (0, r), (-1, r), 0.3, colors.grey))
    t.setStyle(TableStyle(style))
    roman_tab = ROMAN[tab_no[0] - 1]
    story.append(KeepTogether([Paragraph(f"TABLE {roman_tab}<br/>{small_caps(caption)}", TCAPTION), t, Spacer(1, 6)]))
    return roman_tab


def fmt_rate(x):
    x = float(x)
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    return f"{x / 1e3:.0f}k"


# ================================================================ PAPER TEXT
story.append(Paragraph("DATS: Degree-Aware Tiled Search for Fast<br/>Max-Cut Local Search on GPUs", TITLE))
story.append(Table([[Paragraph("Vedant Tamhanekar", AUTHOR), Paragraph("Siddhant Salunke", AUTHOR)],
                    [Paragraph("Veermata Jijabai Technological Institute (VJTI)<br/>Mumbai, India", AFFIL),
                     Paragraph("Veermata Jijabai Technological Institute (VJTI)<br/>Mumbai, India", AFFIL)]],
                   colWidths=[3.4 * inch, 3.4 * inch]))
story.append(FrameBreak())

# ---------------- abstract
if LARGE:
    gpu_abs = (f" On an NVIDIA {GPU_NAME}, DATS is {rng_txt([gspeed(g, 'gpu_dats') for g in LARGE])}× faster than a "
               f"one-thread-per-vertex GPU baseline on the graphs with more than 15k vertices, while degree-aware "
               f"updates alone give almost no gain there. On small G-set graphs the tile bookkeeping costs more than "
               f"it saves, and a plain full scan is faster.")
else:
    gpu_abs = ""
abstract = (
    "<i>Abstract</i>—Local search methods like tabu search are among the best heuristics for the Max-Cut "
    "problem, and earlier work has shown that running many searches in parallel on a GPU gives large speedups. "
    "Most of these GPU codes give one thread to each vertex and are tested on G-set graphs, where all vertices "
    "have almost the same degree. On real power-law graphs this design has two problems. First, when a hub vertex "
    "is flipped, one thread has to update thousands of neighbours while the rest of the block waits. Second, and "
    "less discussed, every step scans all <i>n</i> vertices to find the best move even though a flip changes only "
    "the gains of the flipped vertex and its neighbours. In this paper we present DATS (Degree-Aware Tiled Search), "
    "a GPU tabu search for Max-Cut that addresses both. DATS keeps the best move of every 32-vertex tile in shared "
    "memory and after each flip recomputes only the tiles that changed, using one warp per tile. The neighbour "
    "update is given to one warp or to the whole block depending on the degree of the flipped vertex, and a BFS "
    "relabelling keeps neighbours in the same tiles. The tiled selection is exact, so DATS makes exactly the "
    f"same moves as a full scan. On four SNAP graphs, a 100k-vertex power-law graph and two sparse G-set graphs, tiled "
    f"selection reads {rng_txt(sparse_work, '{:.0f}')}× fewer entries per step and alone gives "
    f"{rng_txt(sparse_tiled_speed)}× speedup on a single CPU core. Together with OpenMP on a "
    f"{num_threads_cpu}-core CPU it reaches up to {max(omp_dats_speed):.0f}× over the serial full-scan "
    f"search.{gpu_abs}"
)
P(abstract, ABSTRACT)
P("<i>Index Terms</i>—Max-Cut, tabu search, GPU, CUDA, load balancing, power-law graphs, Ising model, "
  "incremental selection.", ParagraphStyle("kw", parent=ABSTRACT, spaceBefore=4))
if not gpu_graphs:
    P("[DRAFT: the GPU results (Section VI-E and the GPU sentence of the abstract) are not filled yet. They are "
      "produced automatically after running <font name='Mono' size='7.5'>./run_experiments.sh gpu</font> "
      "on a CUDA machine and rebuilding this PDF.]", PENDING)

# ---------------- I. introduction
section("Introduction")
P("In the Max-Cut problem we are given a graph and we have to split its vertices into two groups so that the total "
  "weight of the edges going between the groups is as large as possible. It is one of Karp's original NP-complete "
  "problems [1], and it shows up in VLSI design, statistical physics and network analysis. Max-Cut is also the same "
  "as finding the ground state of an Ising spin glass [2], which is why it is often used to test Ising machines and "
  "quantum annealers. The best approximation algorithm, by Goemans and Williamson [3], is based on semidefinite "
  "programming and is too slow for big graphs, so in practice heuristics such as Breakout Local Search (BLS) [4] are "
  "used.")
P("Local search is a good fit for GPUs because many independent searches can be run at the same time. Cook et al. [5] "
  "solved Max-Cut with GPU Ising annealing and reported speedups of about 2000× over a CPU on their biggest "
  "graphs, and Yasudo et al. [6] ran thousands of searches in parallel on multiple GPUs. However, in [5] each thread "
  "looks after one vertex, and the authors point out that the threads are not load balanced when vertices have "
  "different degrees, but leave this open. Most of these works are also tested on the G-set [7], whose graphs have "
  "nearly uniform degrees. Real graphs like e-mail, social or co-author networks follow a power-law degree "
  "distribution [24] (Fig. [[fig:degree]]): a few hubs have thousands of neighbours while most vertices have less than ten.")
P("Our first idea was simply to share the neighbour update of a hub among many threads (Fig. [[fig:timeline]]). While building this "
  "we noticed something more important. In a block that runs one tabu search, every step does two things: (1) find "
  "the best allowed move, which scans all <i>n</i> gains, and (2) apply it, which touches only deg(<i>v</i>) "
  "neighbours. On our graphs the average degree is between 4 and 48, while <i>n</i> is up to 100,000, so the scan "
  "costs hundreds of times more than the update. Balancing the update alone therefore cannot give much. The scan is "
  "also mostly wasted work, because only a few gains changed since the last step.")
figure("timeline", "Load imbalance when a hub vertex is flipped. (a) With one thread per vertex, the thread "
       "owning the hub does all updates while the others wait. (b) When the work is shared, all threads finish "
       "at about the same time.")
P("Based on this, the main contributions of this paper are:", BODY)
for b in [
    "<b>Tiled incremental selection.</b> Vertices are grouped into tiles of 32. Each tile keeps its best non-tabu "
    "move, its best tabu move (for aspiration) and the step when its next tabu vertex becomes free. After a flip only "
    "dirty tiles are recomputed, one warp per tile, and the best move is found from <i>n</i>/32 summaries. We prove "
    "that this gives exactly the same move as a full scan.",
    "<b>Degree-aware neighbour updates.</b> The gain update of the flipped vertex is done by one warp (low and medium "
    "degree) or the whole thread block (hubs). We show that this matters more after the selection is made cheap.",
    "<b>BFS relabelling</b> of vertices so that neighbours fall in the same tiles, which reduces the number of dirty "
    "tiles when the input ids are in a bad order.",
    "An evaluation on G-set, SNAP and synthetic power-law graphs with a CPU serial, an OpenMP and four GPU variants "
    "that all follow the <i>identical</i> search path, so speed can be compared fairly and correctness can be checked "
    "exactly.",
]:
    story.append(Paragraph(b, BULLET, bulletText="•"))
P("The rest of the paper is organised as follows. Section II covers background on Max-Cut, GPUs and related "
  "work. Section III analyses the cost of one search step with a small example. Section IV explains DATS and proves "
  "that the tiled selection is exact. Section V gives the experimental setup and Section VI the results, including "
  "scaling, ablation and quality studies. Section VII discusses limitations and Section VIII concludes. The "
  "appendix describes how to reproduce all results.")

# ---------------- II. background
section("Background and Related Work")
subsection("Max-Cut and Local Search")
P("Let <i>G</i> = (<i>V</i>, <i>E</i>) be a graph with <i>n</i> vertices, <i>m</i> edges and weights "
  "<i>w<sub>ij</sub></i>. A solution is a vector <i>x</i> ∈ {0,1}<super><i>n</i></super> and its cut value is", BODY_NI)
equation(EQ["cut"], 1)
P("Local search keeps for every vertex its <i>gain</i>, the change of the cut if <i>v</i> is moved to the other "
  "side:", BODY_NI)
equation(EQ["gain"], 2)
P("When <i>v</i> is flipped, its own gain becomes −<i>g<sub>v</sub></i> and every neighbour <i>u</i> is updated "
  "as", BODY_NI)
equation(EQ["update"], 3)
P("where σ<sub><i>uv</i></sub> is taken before the flip. So the work of a flip is proportional to "
  "deg(<i>v</i>). Tabu search [8] always makes the best move, even if it is negative, and forbids moving a vertex "
  "back for a few steps (its <i>tenure</i>). A tabu move is still allowed if it gives a new best cut "
  "(aspiration). BLS [4] adds a perturbation phase: when no improvement is found for a number of steps, some random "
  "moves are made, and the number of moves grows if the search keeps falling back to the same local optimum. We use "
  "these rules in all our versions.")
subsection("Other Max-Cut Solvers and Ising Machines")
P("Besides BLS, many other heuristics have been proposed for Max-Cut. Burer et al. [16] use a rank-two relaxation "
  "of the semidefinite program, which is much faster than solving the full relaxation and was for a long time one "
  "of the strongest methods on the G-set. Kochenberger et al. [17] show that a tabu search on the QUBO form can "
  "handle Max-Cut instances with up to 10,000 vertices, and Wu and Hao [26] combine tabu search with a memetic "
  "algorithm. All of these are sequential CPU methods. Their core loop is the same as ours: keep the gains, pick "
  "the best move, update the neighbours, so the ideas of this paper can also be used inside them.")
P("Because Max-Cut is the same as finding the ground state of an Ising model [2], it is also a standard benchmark "
  "for special hardware. Examples are the coherent Ising machine built with optical pulses [18], the CMOS "
  "annealing chip of Yamaoka et al. [20], and simulated bifurcation [19], which runs on GPUs and FPGAs. These "
  "machines are fast but need special hardware or update all spins together. Our work stays with ordinary GPUs and "
  "an exact tabu search, and focuses on making each step cheaper.")
subsection("GPU Methods for Max-Cut and QUBO")
P("Cook et al. [5] map Max-Cut to an Ising model and run simulated annealing on the GPU with one thread per spin. "
  "Yasudo et al. propose Adaptive Bulk Search [6], where each CUDA block runs its own local search on a QUBO problem "
  "stored as a dense <i>n</i>×<i>n</i> matrix, guided by a genetic algorithm on the host. Because of the dense "
  "matrix the memory grows as <i>n</i><super>2</super>. Their later work [9] stores the QUBO as a sparse graph to "
  "scale to bigger instances. We also use one block per search and a sparse graph, so these are not our "
  "contribution. As far as we know none of these works changes the number of threads by vertex degree or avoids "
  "the full scan for the best move.")
subsection("GPU Architecture Basics")
P("For readers less familiar with CUDA [25], we briefly describe the parts of the GPU that matter for our design "
  "(Fig. [[fig:gpu_basics]]). A kernel is launched as a grid of thread blocks. Each block runs on one streaming multiprocessor (SM) "
  "and its threads are grouped into warps of 32 threads that execute the same instruction together. The threads "
  "of a block can share data through <i>shared memory</i>, which is small (up to 64 KB per block on the T4) but "
  "much faster than the global memory used by all blocks. Threads in one block can wait for each other with "
  "<font name='Mono' size='7.5'>__syncthreads()</font>, and the 32 lanes of a warp can exchange registers "
  "directly with shuffle instructions such as <font name='Mono' size='7.5'>__shfl_down_sync</font>, which is the "
  "usual way to build fast reductions [29].")
figure("gpu_basics", "The CUDA execution model used in this paper: a grid of independent blocks, warps of 32 "
       "lanes inside a block, fast per-block shared memory and slower global memory.")
P("Two points follow from this. First, if one thread of a block has much more work than the others, the whole "
  "block waits for it at the next synchronisation, which is the load imbalance of Fig. [[fig:timeline]]. Second, warps are the "
  "natural unit of work: 32 values can be reduced inside a warp with five shuffle steps and without touching "
  "shared memory. Both points shape our design. A tile has exactly 32 vertices so that one warp can refresh it, and "
  "the update of a vertex is given to a warp or to the whole block depending on its degree.", BODY)
subsection("Load Balancing and Incremental Gain Structures")
P("Degree imbalance is a well known problem in GPU graph analytics. Merrill et al. [10] process the neighbour list "
  "of a vertex with a thread, a warp or the whole block depending on its size, and Gunrock [11] uses similar "
  "strategies. Our degree-aware update is an adaptation of this idea to local search, where only one vertex is "
  "expanded per step. For finding the best gain fast on a CPU, the Fiduccia–Mattheyses heuristic [12] keeps gain "
  "buckets with linked lists. Bucket lists need pointer updates that are serial and do not map well to warps, and "
  "they do not handle the tabu and aspiration rules directly. Our tile summaries are a flat, warp-friendly "
  "alternative. Hong et al. [21] proposed virtual warp-centric processing, where a group of lanes works on one "
  "vertex, which is close to our medium-degree class.")
P("Vertex reordering to improve locality is also standard. Cuthill–McKee [13] reduces the bandwidth of sparse "
  "matrices, and newer orderings like Gorder [22] and Rabbit Order [23] place vertices that share neighbours close "
  "together to improve cache use in graph analytics. We use a simple BFS version of this idea for a new goal, which "
  "is to reduce the number of dirty tiles after a flip. Better orderings could be plugged in directly.")

# ---------------- III. cost analysis
section("Cost of One Search Step")
P("Table [[tab:notation]] lists the symbols used in the rest of the paper.")
table("Notation", [
    ["Symbol", "Meaning"],
    ["<i>n</i>, <i>m</i>", "number of vertices and edges"],
    ["<i>x<sub>v</sub></i>", "side of vertex <i>v</i> (0 or 1)"],
    ["<i>g<sub>v</sub></i>", "gain of <i>v</i>, change of the cut if <i>v</i> is flipped"],
    ["<i>N</i>(<i>v</i>), deg(<i>v</i>)", "neighbours and degree of <i>v</i>"],
    ["<i>P</i>", "threads per block"],
    ["<i>T</i>", "tile size (32 on the GPU)"],
    ["<i>F<sub>t</sub></i>, <i>T<sub>t</sub></i>", "best non-tabu and best tabu move key of tile <i>t</i>"],
    ["<i>E<sub>t</sub></i>", "first step at which a tabu vertex of tile <i>t</i> becomes free"],
    ["<i>D</i>(<i>v</i>)", "number of different tiles in <i>N</i>(<i>v</i>) \u222a {<i>v</i>}"],
    ["<i>K</i>(<i>v</i>)", "64-bit move key of <i>v</i>, see (4)"],
], [1.05 * inch, 2.35 * inch], key="notation")
subsection("A Small Example")
P(f"Fig. [[fig:example]] shows one step on a small graph with six vertices and nine unit-weight edges. In (a) the "
  f"cut is {ex_cut(EX_SIDE0)} and the gains are computed with (2). The largest gain is "
  f"{EX_G0[EX_FLIP]:+d} at vertex {EX_FLIP}, so a tabu search flips {EX_FLIP}. After the flip (b) the cut is "
  f"{ex_cut(EX_SIDE1)}, the gain of {EX_FLIP} becomes {EX_G1[EX_FLIP]:+d}, and only the gains of its neighbours "
  f"{', '.join(EX_NEIGH[:-1])} and {EX_NEIGH[-1]} change, each by \u00b12 as in (3). The gain of every other "
  "vertex stays the same. Still, a full-scan search looks at all six gains again in the next step to find the "
  "best move. On a graph with 100,000 vertices and average degree 8, one flip changes about 9 gains, but the next "
  "step reads all 100,000 of them. This is the waste we remove in Section IV.")
figure("example", f"One search step on a small graph. Black and white show the two sides, solid edges are cut and "
       f"dashed edges are not. Flipping {EX_FLIP} (the largest gain) changes only the gains of {EX_FLIP} and its "
       "neighbours (red).")
subsection("Where the Time Goes")
P("Consider one tabu search run by a block of <i>P</i> threads. In the usual design (“full scan”), a step "
  "reads all <i>n</i> gains and tabu values to find the best allowed move, so the selection costs "
  "<i>n</i>/<i>P</i> per thread plus a reduction. The update then costs deg(<i>v</i>) if one thread does it "
  "(naive), or about deg(<i>v</i>)/32 or deg(<i>v</i>)/<i>P</i> with degree-aware sharing. Table [[tab:datasets]] shows that on "
  "our power-law graphs a step scans 7,000 to 100,000 entries while the average degree is about 10. Even the largest hub "
  "in powerlaw-100k (degree 16,759) is smaller than <i>n</i>. Also, we measured that hubs (degree above 1024) are "
  f"selected in only {min(hub_pct):.2f}–{max(hub_pct):.2f}% of steps. We think this is because a hub has a "
  "large gain only for a short time, and after it is flipped it stays tabu for many steps. This tells us that the selection is the main cost and has "
  "to be fixed first, and the hub update becomes important only after that. This is the motivation behind "
  "combining the two techniques.")

# datasets table
rows = [["Graph", "Type", "<i>n</i>", "<i>m</i>", "avg deg", "max deg", "hubs"]]
for g in ALL_GRAPHS:
    r = cpu[g]["cpu_full"]
    n, m, md = int(r["n"]), int(r["m"]), int(r["maxdeg"])
    d = DEGREES.get(g)
    hubs = int((d > 1024).sum()) if d is not None else "–"
    typ = GSET_TYPE.get(g, "synthetic, γ=2.1" if g in SYNTH else "SNAP")
    rows.append([g, typ, f"{n:,}", f"{m:,}", f"{2 * m / n:.1f}", f"{md:,}", hubs if g not in SYNTH else "–"])
cw = [0.78, 0.68, 0.48, 0.52, 0.38, 0.42, 0.3]
table("Graphs used in the experiments", rows, [c * inch for c in cw], highlight_rows=(len(GSET) + 1,), key="datasets")

# ---------------- IV. method
section("Proposed Method: DATS")
P("Fig. [[fig:arch]] shows the overall design. The host reads the graph, builds the CSR arrays, relabels the vertices and "
  "computes the degree class of each vertex. The graph is copied once to global memory and shared by all blocks. "
  "Every block runs one independent tabu search and keeps its search state in shared memory.")
figure("arch", "Mapping of DATS on the GPU. All blocks read the same CSR graph from global memory, each block "
       "keeps its own solution, tile summaries and (if they fit) gains in shared memory.")
subsection("Move Keys")
P("To compare moves with a single integer comparison, we pack each candidate move into a 64-bit key:", BODY_NI)
equation(EQ["key"], 4)
P("The upper half is the gain with its sign bit flipped, so that a larger gain gives a larger unsigned number. The "
  "lower half is a tie-break value: φ is the odd constant 0x9E3779B1 and <i>s</i> is a random salt. Because "
  "φ is odd it has an inverse modulo 2<super>32</super>, so the vertex can be recovered from the key. The "
  "salt is changed after every perturbation, so ties are broken in a different random order in each phase. With "
  "this encoding, “best move” is just the maximum key, which works directly with warp shuffle reductions.")
subsection("Tiled Incremental Selection")
P("The vertices are divided into tiles of 32 consecutive ids, one per warp lane. For each tile <i>t</i> the block "
  "stores three values: <i>F<sub>t</sub></i>, the maximum key among the non-tabu vertices; <i>T<sub>t</sub></i>, "
  "the maximum key among the tabu vertices; and <i>E<sub>t</sub></i>, the smallest step at which one of its tabu "
  "vertices becomes free. A tile can only become wrong in two ways: a gain or tabu value inside it changes "
  "(only after a flip of <i>v</i> or a neighbour of <i>v</i>), or the step reaches <i>E<sub>t</sub></i>. So during "
  "the neighbour update each thread also sets the bit of the neighbour's tile in a dirty bitmask using "
  "<font name='Mono' size='7.5'>atomicOr</font>. After the flip, the warps go over the bitmask and every dirty tile "
  "is recomputed by one warp, where lane <i>i</i> reads vertex 32<i>t</i>+<i>i</i> and three shuffle reductions "
  "give <i>F<sub>t</sub></i>, <i>T<sub>t</sub></i> and <i>E<sub>t</sub></i> (Fig. [[fig:tiles]]).")
figure("tiles", "Tiled incremental selection. After flipping <i>v</i>, only the tiles that hold <i>v</i> or its "
       "neighbours are recomputed. The next move is the maximum over the tile summaries.")
P("To choose a move, the threads go over the tile summaries. If <i>E<sub>t</sub></i> ≤ step, the tile is "
  "refreshed first (this is rare, at most one vertex becomes free per step on average). The candidate of a tile is "
  "<i>F<sub>t</sub></i>, or <i>T<sub>t</sub></i> if <i>T<sub>t</sub></i> passes the aspiration test and is larger. "
  "A block reduction gives the global maximum.")
P("<b>Exactness.</b> We now show that the tiled selection always returns the same move as a full scan. Let "
  "<i>A</i>(<i>s</i>) be the set of vertices allowed at step <i>s</i>, i.e. the non-tabu vertices plus the tabu "
  "vertices that pass the aspiration test cut + <i>g<sub>v</sub></i> > best.", BODY_NI)
P("<b>Lemma 1.</b> <i>Let the summary of tile t be computed at step s<sub>0</sub>. If no vertex of t has its gain "
  "or tabu value changed and the salt is not changed, then for every step s with s<sub>0</sub> \u2264 s &lt; "
  "E<sub>t</sub> the values F<sub>t</sub> and T<sub>t</sub> are the largest keys among the non-tabu and the tabu "
  "vertices of t at step s.</i>", BODY_NI)
P("<i>Proof.</i> A vertex is tabu at step <i>s</i> if its tabu value is larger than <i>s</i>. A vertex that was "
  "free at <i>s</i><sub>0</sub> stays free, since its tabu value did not change. A vertex that was tabu at "
  "<i>s</i><sub>0</sub> has a tabu value of at least <i>E<sub>t</sub></i> > <i>s</i>, so it is still tabu. So the "
  "two groups are the same as at <i>s</i><sub>0</sub>. The keys depend only on the gain, the vertex id and the "
  "salt, which did not change, so the maxima are the same. \u220e", BODY_NI)
P("<b>Theorem 1.</b> <i>At every step, the move chosen by the tiled selection is the vertex with the largest key "
  "in A(s), i.e. the same move as a full scan.</i>", BODY_NI)
P("<i>Proof.</i> A gain or tabu value changes only for the flipped vertex and its neighbours, and all their tiles "
  "are marked dirty and refreshed before the next step. Tiles with <i>E<sub>t</sub></i> \u2264 <i>s</i> are "
  "refreshed during selection, and the salt changes only in a perturbation, after which all tiles are rebuilt. So "
  "by Lemma 1 every summary is correct when it is read. For one tile, the best allowed vertex is either the best "
  "non-tabu vertex <i>F<sub>t</sub></i>, or a tabu vertex that passes the aspiration test. The aspiration test "
  "cut + <i>g</i> > best is monotone in <i>g</i>, and a larger key always has a larger or equal gain. So if any "
  "tabu vertex of the tile passes the test, the one with the largest key <i>T<sub>t</sub></i> passes it too and "
  "is the best tabu candidate. Taking the larger of the two candidates for every tile and then the maximum over "
  "all tiles gives the maximum key over <i>A</i>(<i>s</i>). \u220e", BODY_NI)
P("Because the keys are unique (the low half is a bijection of the vertex id), there are no ties, and the result "
  "does not depend on the order in which the threads look at the tiles. We also check Theorem 1 experimentally in "
  "Section VI-A.")
P("<b>Cost.</b> Let <i>D</i>(<i>v</i>) be the number of different tiles in <i>N</i>(<i>v</i>) ∪ {<i>v</i>} and "
  "<i>X<sub>t</sub></i> the number of expired tiles in a step. The entries read per step are", BODY_NI)
equation(EQ["cost"], 5)
P("instead of <i>n</i>. Since <i>D</i>(<i>v</i>) ≤ deg(<i>v</i>)+1, the selection is never worse than a full "
  "scan by more than a constant, and for sparse graphs it is much smaller. For dense graphs where <i>D</i>(<i>v</i>) "
  "is close to <i>n</i>/32 there is no gain, which we also see in the results. Table [[tab:cost]] summarises the "
  "cost of one step for the main versions.")
table("Cost of one step for one search (P threads per block)", [
    ["", "Naive", "DA", "DATS"],
    ["selection reads", "<i>n</i>", "<i>n</i>", "<i>n</i>/32 + 32<i>D</i>(<i>v</i>)"],
    ["selection time", "<i>n</i>/<i>P</i>", "<i>n</i>/<i>P</i>", "<i>n</i>/(32<i>P</i>) + <i>D</i>(<i>v</i>)\u00b732/<i>P</i>"],
    ["update time", "deg(<i>v</i>)", "\u2248deg(<i>v</i>)/32 or /<i>P</i>", "\u2248deg(<i>v</i>)/32 or /<i>P</i>"],
    ["extra state", "\u2013", "\u2013", "20 B per tile"],
    ["exact BLS moves", "yes", "yes", "yes"],
], [0.8 * inch, 0.55 * inch, 0.95 * inch, 1.15 * inch], key="cost")
subsection("Degree-Aware Neighbour Update")
P("For the update in (3), the vertices are divided into three classes on the host. If deg(<i>v</i>) ≤ 32, warp 0 "
  "updates all neighbours in one go, one per lane. If 32 < deg(<i>v</i>) ≤ 1024, warp 0 loops over the "
  "list 32 neighbours at a time. Above 1024, all warps of the block share the list. Every neighbour appears once in "
  "the CSR row, so no two threads write the same gain and no atomics are needed for the gains. The low and medium "
  "classes use only one warp so that the other warps are not involved for short lists, which keeps the update "
  "simple. Tuning these limits is left for future work.")
subsection("BFS Relabelling")
P("The number of dirty tiles depends on how the neighbours of <i>v</i> are spread over the ids. We relabel the "
  "vertices with a breadth-first search that starts from the vertex of highest degree, so the vertices found from "
  "the same parent get consecutive ids. Unreached components start from the next highest degree vertex. The "
  "relabelling is done once on the host in <i>O</i>(<i>n</i> + <i>m</i>) time.")
subsection("GPU Implementation")
P("The shared memory of a block holds, in this order, the tile summaries (20 bytes per tile), the dirty bitmask, "
  "the solution (1 bit per vertex) and the gains (4 bytes per vertex) if they still fit, otherwise the gains stay in "
  "a global memory slice of the block. The tile structure costs only 0.625 bytes per vertex, so with 64 KB of shared "
  "memory graphs with about 80k vertices keep everything except the gains on chip. The tabu array is kept in "
  "global memory. Thread 0 owns the random number generator, the current and best cut and the perturbation state. "
  "The complete kernel step is given in Algorithm 1. All our versions use the same xorshift generator [28] and "
  "seeds, so block <i>i</i> on the GPU and search <i>i</i> on the CPU are designed to make exactly the same moves.")
mem = {g: shared_layout(int(cpu[g]["cpu_full"]["n"])) for g in ["G55", "email-Enron", "soc-Epinions1"] if g in cpu}
P(f"Fig. [[fig:memlayout]] shows how much shared memory a block needs on a T4 for some of our graphs. For small "
  f"graphs like G55 everything fits, including the gains "
  f"({sum(mem['G55'][0].values()) / 1024:.1f} KB). For email-Enron the gains alone would need "
  f"{4 * int(cpu['email-Enron']['cpu_full']['n']) / 1024:.0f} KB, so they stay in global memory, and only the "
  f"tiles, bitmask and solution are on chip ({sum(mem['email-Enron'][0].values()) / 1024:.1f} KB). Even "
  f"soc-Epinions1 with 76k vertices needs only {sum(mem['soc-Epinions1'][0].values()) / 1024:.1f} KB for these "
  "parts, which leaves room for two blocks per SM.")
figure("memlayout", "Shared memory used per block on the T4 (64 KB limit, minus 2 KB we keep for static "
       "variables). Gains are kept in shared memory only when everything fits.")
algo = """for each block b in parallel:            // one search per block
  x <- random bits(seed_b); compute gains g
  s <- rand(); build all tile summaries
  for step = 1 .. T:
    for each tile t (strided over threads):
      if E[t] <= step: refresh(t)
      c <- F[t]
      if T[t] aspirates and T[t] > c: c <- T[t]
    v <- vertex of block_max(c)       // shuffle
    if deg(v) <= 32:   warp 0, 1 neighbour/lane
    elif deg <= 1024:  warp 0 loops over N(v)
    else:              all warps share N(v)
       (update g[u], atomicOr dirty[tile(u)])
    flip x[v]; g[v] <- -g[v]; make v tabu
    each warp: refresh its dirty tiles
    if stuck: random flips, s <- rand(),
              rebuild all tiles
  write best cut and solution of block b"""
algo_tbl = Table([[Paragraph("<b>Algorithm 1:</b> DATS kernel (one block)", ParagraphStyle("at", parent=BODY_NI, fontSize=8))],
                  [Preformatted(algo, CODE)]], colWidths=[COL_W_IN * inch])
algo_tbl.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                              ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                              ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                              ("LEFTPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 2)]))
story.append(KeepTogether([Spacer(1, 4), algo_tbl, Spacer(1, 6)]))
P("Algorithm 2 shows the tile refresh in more detail. The dirty bitmask has one bit per tile, so it needs only "
  "<i>n</i>/256 bytes. Each warp takes some 32-bit words of the mask and uses "
  "<font name='Mono' size='7.5'>__ffs</font> (find first set bit) to jump directly to the dirty tiles, so clean "
  "tiles cost nothing. Inside a tile, lane <i>i</i> handles vertex 32<i>t</i>+<i>i</i>, and three shuffle "
  "reductions of five steps each give the two maxima and the minimum expiry. No shared memory atomics are needed "
  "here, because each tile is refreshed by exactly one warp.")
algo2 = """refresh_dirty(next_step):              // all warps
  for w = warp_id; w < n_words; w += n_warps:
    bits <- dirty[w]
    while bits != 0:
      b <- ffs(bits) - 1; bits <- bits & (bits-1)
      t <- 32*w + b;  v <- 32*t + lane
      k <- key(g[v], v, s)
      if tabu_until[v] > next_step:
           free <- 0;  tabu <- k;  exp <- tabu_until[v]
      else free <- k;  tabu <- 0;  exp <- INF
      for off in 16, 8, 4, 2, 1:          // shuffles
        free <- max(free, shfl_down(free, off))
        tabu <- max(tabu, shfl_down(tabu, off))
        exp  <- min(exp,  shfl_down(exp,  off))
      if lane == 0: F[t], T[t], E[t] <- free, tabu, exp
    if lane == 0: dirty[w] <- 0"""
algo2_tbl = Table([[Paragraph("<b>Algorithm 2:</b> refreshing the dirty tiles", ParagraphStyle("at2", parent=BODY_NI, fontSize=8))],
                   [Preformatted(algo2, CODE)]], colWidths=[COL_W_IN * inch])
algo2_tbl.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                               ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                               ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                               ("LEFTPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 2)]))
story.append(KeepTogether([Spacer(1, 4), algo2_tbl, Spacer(1, 6)]))

# ---------------- V. setup
section("Experimental Setup")
P(f"<b>Hardware.</b> The CPU experiments were run on an {CPU_NAME} machine with {num_threads_cpu} cores. "
  + (f"The GPU experiments were run on an NVIDIA {GPU_NAME} (40 SMs, 16 GB) on Google Colab. " if HAVE_GPU else
     "<font color='#c53030'>[GPU model will be filled after the GPU run.]</font> ") +
  "All code is C++17 with OpenMP and CUDA, compiled with -O3, and is in a single source file.", BODY_NI)
P("<b>Graphs.</b> Table [[tab:datasets]] lists the graphs. From the G-set [7] we use four random graphs (G1, G22, G43, G55), "
  "a planar graph (G14) and a toroidal grid (G48). From SNAP [14] we use wiki-Vote, email-Enron, ca-AstroPh and "
  "soc-Epinions1 as undirected, unweighted graphs. We also generate a Chung–Lu power-law graph [15] with 100k "
  "vertices, exponent 2.1 and average degree 8, which has much bigger hubs. Fig. [[fig:degree]] shows the degree distributions.")
figure("degree", "Degree distribution (complementary CDF) of the SNAP graphs compared with the G-set graph G55. "
       "Dotted lines are the class limits 32 and 1024.")
P(f"<b>Parameters.</b> Unless stated, each version runs {SEARCHES} searches of {STEPS:,} steps. The tabu tenure is "
  "3 + rand(0, <i>n</i>/10), perturbation starts after 500 non-improving steps, and its strength goes from "
  "max(2, <i>n</i>/100) up to <i>n</i>/20, capped at 100 and 500. On the GPU we use 256 threads per block and two "
  "blocks per SM.", BODY_NI)
P("<b>Versions.</b> On the CPU: <i>full</i> (serial full scan), <i>tiled</i> (serial, tiled selection), "
  "<i>tiled+BFS</i>, and the OpenMP versions of full and tiled+BFS, which run the searches in parallel. On the GPU: "
  "<i>Naive</i> (full scan, one thread per vertex update, as in [5]), <i>DA</i> (full scan, degree-aware update), "
  "<i>DA+T</i> (degree-aware + tiles), <i>T+R</i> (tiles + relabel with naive update) and <i>DATS</i> (all three). "
  "We report moves per second (including perturbation moves), the work of the selection (entries read per step), "
  "dirty tiles per step and the best cut. Every best cut reported by a version is recomputed from its solution "
  "with an independent function.", BODY_NI)
P("<b>Implementation and validation.</b> All versions are in one C++/CUDA source file of about 1,600 lines. The "
  "CPU versions use OpenMP [27] with a dynamic schedule, one search per iteration. The GPU part is compiled only "
  "when the file is built with nvcc, so the same file also builds as a CPU-only program. Because all versions use "
  "the same xorshift generator [28] and the same seeds, and the tiled selection is exact (Theorem 1), a version can "
  "be checked against another one by comparing the best cut of every search, not just the overall best. The "
  "program prints these checks after every run, and a run that fails them is treated as a bug. All tables and "
  "figures in this paper are generated by a script directly from the CSV files written by the program, so no "
  "number is copied by hand. The appendix lists the exact commands.", BODY_NI)
P("<b>Timing.</b> CPU times are wall-clock times measured with <font name='Mono' size='7.5'>std::chrono</font> "
  "around all searches. GPU times are measured with CUDA events around the kernel only, after one untimed warm-up "
  "launch, and do not include copying the graph (which is done once). Because many G-set runs take only a few "
  "milliseconds, small differences on those graphs should not be over-interpreted.", BODY_NI)

# ---------------- VI. results
section("Results")
subsection("Correctness")
same_ok = all(cpu[g]["cpu_full"]["best_cut"] == cpu[g]["cpu_tiled"]["best_cut"] for g in cpu)
P(f"For all {len(cpu)} graph inputs (including the shuffled ones), the serial tiled version found the same best cut "
  f"as the full scan for every one of the {SEARCHES} searches{'' if same_ok else ' (NOT TRUE, check!)'}, which confirms "
  "the exactness argument of Section IV. All reported cuts matched the recomputed value. "
  + ("On the GPU, all variants that use the same vertex order found the same best cut on every graph, and all "
     "GPU cuts matched the recomputed value." if HAVE_GPU else ""))

subsection("Selection Work")
work_fig = figure("work", "Entries read per step to choose a move (log scale), full scan versus tiled "
                  "selection. The number on top is the reduction.")
P(f"Fig. {work_fig} shows the average number of entries read per step. On the sparse graphs the tiled selection "
  f"reads {rng_txt(sparse_work, '{:.0f}')}× fewer entries. For email-Enron, for example, it reads about "
  f"{f(cpu['email-Enron']['cpu_tiled'], 'sel_work_per_step'):,.0f} entries instead of 36,692. Most of what is "
  "left is the scan over the <i>n</i>/32 summaries. On the dense random graphs G1 and G43 a flip makes "
  f"{f(cpu['G1']['cpu_tiled'], 'dirty_tiles_per_step'):.1f} of 25 tiles and "
  f"{f(cpu['G43']['cpu_tiled'], 'dirty_tiles_per_step'):.1f} of 32 tiles dirty, so the reduction is less than 2×.")

subsection("Where the Selection Work Goes")
scan_share = {g: 100 * work_parts(g)[0] / sum(work_parts(g)) for g in SPARSE}
dirty_share = {g: 100 * work_parts(g)[1] / sum(work_parts(g)) for g in SPARSE}
big_sparse = [g for g in SPARSE if int(cpu[g]["cpu_full"]["n"]) >= 15000]
P(f"Equation (5) splits the work of the tiled selection into three parts: the scan over the <i>n</i>/32 summaries, "
  f"the refresh of dirty tiles, and the refresh of expired tiles plus the full rebuilds after each perturbation. "
  f"Fig. [[fig:breakdown]] shows the measured split. On the large graphs the summary scan is "
  f"{rng_txt([scan_share[g] for g in big_sparse], '{:.0f}')}% of the work and the dirty refresh "
  f"{rng_txt([dirty_share[g] for g in big_sparse], '{:.0f}')}%. On the smaller sparse graphs like G48 the dirty "
  f"refresh is a larger part ({dirty_share['G48']:.0f}%). This tells us where to look next: on large graphs a "
  "second level of summaries would remove most of the remaining work, while on small graphs the dirty refresh and "
  "the per-perturbation rebuilds matter more.")
figure("breakdown", "Split of the tiled selection work into the scan of tile summaries, the refresh of dirty "
       "tiles, and expired tiles plus rebuilds after perturbations (serial tiled CPU version).")

subsection("CPU Speedup")
cpu_fig = figure("cpu", f"Speedup in moves per second over the serial full-scan search ({SEARCHES} searches, "
                 f"{STEPS:,} steps each, {num_threads_cpu} OpenMP threads).")
P(f"Fig. {cpu_fig} shows that the lower selection work also gives real time savings. On one core, tiled selection "
  f"is {rng_txt(sparse_tiled_speed)}× faster on the sparse graphs. Running the searches in parallel with OpenMP "
  f"gives {rng_txt(omp_full_speed)}× for the full scan, and the two together give "
  f"{rng_txt(omp_dats_speed)}×, i.e. the gains multiply. On G1 and G43 there is no gain from tiling, "
  f"and on G1 the OpenMP tiled version was even a bit slower ({speedup('G1', 'cpu_tiled_bfs_omp'):.1f}× vs "
  f"{speedup('G1', 'cpu_full_omp'):.1f}×) because of the extra bookkeeping. Note that these G-set runs take "
  "only a few milliseconds, so small differences there are within timing noise.")

if THR:
    subsection("Scaling with OpenMP Threads")
    tmax = max(THREAD_COUNTS)
    P(f"Fig. [[fig:omp]] shows how the OpenMP versions scale on email-Enron and soc-Epinions1 when 10 searches are "
      f"shared among 1 to {tmax} threads. At {tmax} threads the full scan reaches "
      f"{omp_speedup('email-Enron', 'cpu_full_omp', tmax):.1f}× and "
      f"{omp_speedup('soc-Epinions1', 'cpu_full_omp', tmax):.1f}×, and the tiled version "
      f"{omp_speedup('email-Enron', 'cpu_tiled_bfs_omp', tmax):.1f}× and "
      f"{omp_speedup('soc-Epinions1', 'cpu_tiled_bfs_omp', tmax):.1f}×. The searches are independent, so the scaling is "
      "limited mainly by the hardware and by the uneven split of 10 searches: with 6 or 8 threads the searches still "
      "need two rounds, which is why these two points give almost the same speed. The tiled version scales a little "
      "less than the full scan. We think this is because each of its steps is much shorter and does more scattered "
      "memory accesses, so the memory system becomes the limit earlier, but we did not measure this. In absolute terms it is still much faster at every thread count, e.g. "
      f"{f(THR['email-Enron-omp1']['cpu_tiled_bfs_omp'], 'moves_per_s') / f(THR['email-Enron-omp1']['cpu_full_omp'], 'moves_per_s'):.1f}× "
      f"with 1 thread and "
      f"{f(THR[f'email-Enron-omp{tmax}']['cpu_tiled_bfs_omp'], 'moves_per_s') / f(THR[f'email-Enron-omp{tmax}']['cpu_full_omp'], 'moves_per_s'):.1f}× "
      f"with {tmax} threads on email-Enron.")
    figure("omp", "OpenMP strong scaling (10 searches of 10,000 steps) on email-Enron (dashed) and soc-Epinions1 "
           "(solid). The dotted line is ideal scaling.")

if SCALE:
    subsection("Scaling with Graph Size")
    ns = [n for n in SCALE_N if all(v in SCALE[f"pl-{n}"] for v in ("cpu_full", "cpu_tiled", "cpu_tiled_bfs"))]

    def sc(n, v, key):
        return f(SCALE[f"pl-{n}"][v], key)

    sp = [sc(n, "cpu_full", "time_s") / sc(n, "cpu_tiled", "time_s") for n in ns]
    wr = [sc(n, "cpu_full", "sel_work_per_step") / sc(n, "cpu_tiled", "sel_work_per_step") for n in ns]
    n0, n1 = ns[0], ns[-1]
    P(f"To see how the method behaves as graphs grow, we generated Chung–Lu graphs with average degree 8 and "
      f"{n0:,} to {n1:,} vertices (the largest hub grows from {int(SCALE[f'pl-{n0}']['cpu_full']['maxdeg']):,} to "
      f"{int(SCALE[f'pl-{n1}']['cpu_full']['maxdeg']):,} neighbours). Fig. [[fig:scaling]](a) shows the time per move "
      f"of the serial versions. The full scan grows linearly with <i>n</i>, as expected. The tiled versions also "
      f"grow, because of the <i>n</i>/32 summary scan, but they stay {rng_txt(sp)}× faster, and the reduction in "
      f"selection work grows from {wr[0]:.0f}× to {wr[-1]:.0f}×. Fig. [[fig:scaling]](b) shows that the number of "
      f"dirty tiles per step also grows with <i>n</i>, because the hubs get bigger. Here BFS relabelling helps more "
      f"for larger graphs: at {n1:,} vertices it reduces the dirty tiles from "
      f"{sc(n1, 'cpu_tiled', 'dirty_tiles_per_step'):.1f} to {sc(n1, 'cpu_tiled_bfs', 'dirty_tiles_per_step'):.1f} "
      f"per step ({100 * (1 - sc(n1, 'cpu_tiled_bfs', 'dirty_tiles_per_step') / sc(n1, 'cpu_tiled', 'dirty_tiles_per_step')):.0f}%), "
      f"while at {n0:,} the difference is only "
      f"{100 * (1 - sc(n0, 'cpu_tiled_bfs', 'dirty_tiles_per_step') / sc(n0, 'cpu_tiled', 'dirty_tiles_per_step')):.0f}%.")
    figure("scaling", "Scaling with graph size on Chung–Lu power-law graphs (average degree 8, serial CPU, 4 "
           "searches). (a) Time per move. (b) Dirty tiles per step for the tiled versions.")

if TILE:
    subsection("Choice of Tile Size")

    def best_t(g):
        return min((t for t in TILE_SIZES if f"{g}-T{t}" in TILE), key=lambda t: f(TILE[f"{g}-T{t}"]["cpu_tiled"], "time_s"))

    P("On the GPU the tile size is fixed to 32 by the warp size, but on the CPU we can change it and check whether "
      "32 is a good choice. With tile size <i>T</i>, the work per step is roughly", BODY_NI)
    equation(EQ["tile"], 6)
    P("where <i>D<sub>T</sub></i>(<i>v</i>) is the number of size-<i>T</i> tiles touched by a flip. Small tiles make "
      "the summary scan long, and large tiles make every dirty tile expensive, so there should be a best value in "
      f"between. Fig. [[fig:tilesize]] confirms this. The selection work and the run time are both U-shaped in <i>T</i>. "
      "The fastest tile size was " + ", ".join(f"{best_t(g)} for {g}" for g in TILE_GRAPHS) + ". "
      "So 32 is the best choice for the smaller graphs, while the large graphs prefer larger tiles: on "
      + ", ".join(f"{g} the best size is {f(TILE[f'{g}-T32']['cpu_tiled'], 'time_s') / f(TILE[f'{g}-T{best_t(g)}']['cpu_tiled'], 'time_s'):.2f}× faster than 32"
                  for g in TILE_GRAPHS if best_t(g) != 32) +
      ". This matches the work breakdown of Fig. [[fig:breakdown]], where the summary scan dominates on large graphs. "
      "On the GPU we keep 32 because one warp refreshes one tile, but a second level of summaries, or two warps per "
      "tile, could give the same benefit there.", BODY_NI)
    figure("tilesize", "Effect of the tile size on the serial tiled CPU search. (a) Entries read per step. (b) Run "
           "time relative to <i>T</i> = 32.")

subsection("Effect of Relabelling")
rr = figure("reorder", "Dirty tiles per step for the tiled CPU search when vertex ids are as in the SNAP file, "
            "randomly shuffled, and shuffled then BFS relabelled.")
P(f"The SNAP files already have a fairly good order, because ids were given while crawling, so BFS relabelling "
  "changes little there. To test the case where the ids are arbitrary, we randomly shuffled the ids before running. "
  f"As Fig. {rr} shows, shuffling increases the dirty tiles a lot, and BFS relabelling brings them back close to "
  f"the level of the original order, a reduction of {rng_txt([100 * x for x in reorder_gain], '{:.0f}')}% compared "
  "with the shuffled order. It does not beat the crawl order of the SNAP files on every graph, so a better "
  "relabelling (for example community based) could still help. On the G-set graphs BFS relabelling changed the "
  f"dirty tiles by at most {max(gset_reorder):.0f}%.")

subsection("GPU Performance")
if gpu_graphs:
    rows = [["Graph"] + [GPU_LABEL[v] for v in GPU_VERS]]
    for g in gpu_graphs:
        rows.append([g] + [fmt_rate(gpu[g][v]["moves_per_s"]) for v in GPU_VERS])
    gt = table(f"GPU throughput in moves per second ({GPU_NAME}, {gpu[gpu_graphs[0]]['gpu_dats']['searches']} blocks)",
               rows, [0.95 * inch] + [0.5 * inch] * 5, highlight_rows=(len(SMALL) + 1,), key="gpu")
    gf = figure("gpu", "Speedup of the GPU variants over the naive GPU version (log scale). Naive: full scan, "
                "one thread updates; DA: full scan, degree-aware update; DA+T: with tiles; T+R: tiles and relabelling "
                "with naive update; DATS: all three.")
    da_full = [gspeed(g, "gpu_degree_full") for g in LARGE]
    da_tiled = [f(gpu[g]["gpu_dats"], "moves_per_s") / f(gpu[g]["gpu_naive_tiled_bfs"], "moves_per_s") for g in gpu_graphs]
    small_ratio = [f(gpu[g]["gpu_dats"], "moves_per_s") / f(gpu[g]["gpu_degree_full"], "moves_per_s") for g in SMALL]
    small_da = [gspeed(g, "gpu_degree_full") for g in SMALL]

    def dats_vs_da(g):
        return f(gpu[g]["gpu_dats"], "moves_per_s") / f(gpu[g]["gpu_degree_full"], "moves_per_s")

    P(f"Table {gt} and Fig. {gf} give the GPU results. The behaviour is very different for large and small graphs, so "
      "we discuss them separately.")
    P(f"<b>Large graphs.</b> On the four graphs with more than 15k vertices, degree-aware updates with the full scan "
      f"change the speed by only {rng_txt(da_full, '{:.2f}')}×, which is what the cost analysis of Section III "
      f"predicts: the scan dominates and balancing the update does not matter. Tiling removes this bottleneck, and DATS "
      f"is {rng_txt([gspeed(g, 'gpu_dats') for g in LARGE])}× faster than the naive version. Once tiling is used, the "
      f"degree-aware update itself gives {rng_txt(da_tiled, '{:.2f}')}× (DATS versus T+R, over all graphs), because "
      "the update is now a much bigger part of a step. So the two techniques are useful together, but degree-aware "
      "updates alone are not.", BODY_NI)
    P(f"<b>Small graphs.</b> On the G-set graphs with at most 2000 vertices the picture is reversed. Here the full "
      f"scan reads only {min(int(gpu[g]['gpu_dats']['n']) for g in SMALL) // 256}–"
      f"{max(int(gpu[g]['gpu_dats']['n']) for g in SMALL) // 256} entries per thread, which is very cheap, and DA "
      f"is already {rng_txt(small_da, '{:.1f}')}× faster than naive. The tile refresh adds extra synchronisation and "
      f"atomic operations to every step, so DATS reaches only {rng_txt(small_ratio, '{:.2f}')}× of the DA speed. The "
      f"crossover is somewhere between a few thousand and ten thousand vertices: on G48 (3000 vertices) DATS gets "
      f"{dats_vs_da('G48'):.2f}× of DA, on wiki-Vote (7115) {dats_vs_da('wiki-Vote'):.2f}×, and on G55 (5000, very "
      f"sparse) {dats_vs_da('G55'):.2f}×. For small dense graphs a full scan is the better choice.", BODY_NI)
    if FIG.get("gpu_threads"):
        tf = figure("gpu_threads", "Throughput on soc-Epinions1 for different block sizes.")
        trows = [r for r in gpu_thread_rows if r["version"] == "gpu_dats"]
        best_t = max(trows, key=lambda r: f(r, "moves_per_s"))
        P(f"Fig. {tf} shows the effect of the block size on soc-Epinions1. DATS is fastest with "
          f"{best_t['graph'].split('tpb')[1]} threads per block. We think larger blocks help the hub updates and the tile scan "
          "but make every __syncthreads more expensive.", BODY_NI)
    reo = [g for g in SNAP if g in gpu and g + "-shuffled" in gpu]
    if reo:
        rows = [["Graph", "file ids: DA+T", "file ids: DATS", "random ids: DA+T", "random ids: DATS"]]
        for g in reo:
            rows.append([g] + [fmt_rate(gpu[h][v]["moves_per_s"]) for h in (g, g + "-shuffled")
                               for v in ("gpu_degree_tiled", "gpu_dats")])
        table("Effect of BFS relabelling on the GPU (moves/s)", rows,
              [0.85 * inch, 0.62 * inch, 0.62 * inch, 0.66 * inch, 0.66 * inch], key="gpu_reorder")

        def rel(h):
            return f(gpu[h]["gpu_dats"], "moves_per_s") / f(gpu[h]["gpu_degree_tiled"], "moves_per_s")

        P(f"Table [[tab:gpu_reorder]] repeats the relabelling study on the GPU. DA+T and DATS differ only in the BFS "
          f"relabelling. With the ids from the SNAP files, relabelling changes the speed by "
          f"{rng_txt([rel(g) for g in reo], '{:.2f}')}×, so it neither helps nor hurts much. With random ids it "
          f"gives {rng_txt([rel(g + '-shuffled') for g in reo], '{:.2f}')}×, and DATS on random ids is almost as fast "
          "as on the original ids. So on the GPU, too, relabelling mainly protects against a bad input order.", BODY_NI)
    # ---------------- CPU versus GPU
    cmp_graphs = [g for g in gpu_graphs if all(v in cpu[g] for v in ("cpu_full", "cpu_full_omp", "cpu_tiled_bfs_omp"))]
    if cmp_graphs:
        subsection("CPU versus GPU")
        CPU_BEST = ("cpu_full_omp", "cpu_tiled_bfs_omp")
        GPU_BEST = ("gpu_naive_full", "gpu_degree_full", "gpu_dats")
        GPU_SHORT = {"gpu_naive_full": "Naive", "gpu_degree_full": "DA", "gpu_dats": "DATS"}

        def best_of(table_, g, versions):
            return max(versions, key=lambda v: f(table_[g][v], "moves_per_s"))

        def mrate(row):
            return f"{f(row, 'moves_per_s') / 1e6:.2f}"

        cmp = {}
        rows = [["Graph", "CPU 1 core", f"CPU best ({num_threads_cpu} cores)", "GPU best", "GPU / CPU best",
                 "GPU / 1 core"]]
        for g in cmp_graphs:
            bc, bg = best_of(cpu, g, CPU_BEST), best_of(gpu, g, GPU_BEST)
            r_best = f(gpu[g][bg], "moves_per_s") / f(cpu[g][bc], "moves_per_s")
            r_one = f(gpu[g][bg], "moves_per_s") / f(cpu[g]["cpu_full"], "moves_per_s")
            cmp[g] = (bc, bg, r_best, r_one)
            rows.append([g, mrate(cpu[g]["cpu_full"]), mrate(cpu[g][bc]) + ("*" if bc == "cpu_tiled_bfs_omp" else ""),
                         f"{mrate(gpu[g][bg])} ({GPU_SHORT[bg]})", f"{r_best:.1f}×", f"{r_one:.0f}×"])
        ct = table("CPU versus GPU throughput (million moves/s)", rows,
                   [0.82 * inch, 0.45 * inch, 0.6 * inch, 0.72 * inch, 0.45 * inch, 0.42 * inch],
                   highlight_rows=(len([g for g in cmp_graphs if g not in LARGE]) + 1,), key="cpu_gpu")
        large_c = [g for g in cmp_graphs if g in LARGE]
        other_c = [g for g in cmp_graphs if g not in LARGE]
        n_tiled_cpu = sum(1 for g in cmp_graphs if cmp[g][0] == "cpu_tiled_bfs_omp")
        da_best = [g for g in other_c if cmp[g][1] == "gpu_degree_full"]
        naive_vs_omp = [f(gpu[g]["gpu_naive_full"], "moves_per_s") / f(cpu[g]["cpu_full_omp"], "moves_per_s")
                        for g in large_c]
        gpu_better_cut = all(max(int(gpu[g][v]["best_cut"]) for v in GPU_BEST) >=
                             max(int(cpu[g][v]["best_cut"]) for v in ("cpu_full", "cpu_full_omp", "cpu_tiled_bfs_omp"))
                             for g in cmp_graphs)
        P(f"Table {ct} compares the CPU and the GPU directly. For each graph we take the fastest CPU version (full scan "
          "or tiled + BFS, both with OpenMP; * marks the tiled one) and the fastest GPU version. The GPU is faster than "
          f"the best CPU version on all {len(cmp_graphs)} graphs. On the large graphs it is "
          f"{rng_txt([cmp[g][2] for g in large_c])}× faster than all {num_threads_cpu} CPU cores and "
          f"{rng_txt([cmp[g][3] for g in large_c], '{:.0f}')}× faster than one core. On the smaller graphs the gap is "
          f"{rng_txt([cmp[g][2] for g in other_c])}×.", BODY_NI)
        P(f"The table also shows that the best method depends on the graph size on both devices. On the large graphs "
          f"the naive GPU version is only {rng_txt(naive_vs_omp)}× faster than the OpenMP full scan, because both "
          "spend most of their time scanning all vertices. Tiling is what opens the gap, and it helps the CPU as well: "
          f"the tiled CPU version is the fastest CPU version on {n_tiled_cpu} of {len(cmp_graphs)} graphs. On the GPU, "
          f"the simple degree-aware full scan (DA) is the fastest version on {len(da_best)} of the {len(other_c)} "
          f"graphs with at most {max(int(cpu[g]['cpu_full']['n']) for g in other_c):,} vertices, and DATS is the "
          "fastest on all large graphs.")
        P(f"In solution quality the GPU found an equal or better cut than the CPU on "
          f"{'every' if gpu_better_cut else 'most'} graph. This comes from running {gpu[cmp_graphs[0]]['gpu_dats']['searches']} "
          f"searches instead of {SEARCHES}, not from a better search, since one search makes the same moves on both "
          f"devices. Also, the CPU ({CPU_NAME}) and the GPU ({GPU_NAME}) are very different machines, so these ratios "
          "are only a rough comparison. The comparisons inside one device (Figs. [[fig:cpu]] and [[fig:gpu]]) are the main results of "
          "this paper.")
else:
    P("[GPU RESULTS PENDING: this subsection, its table and figure are generated from results/gpu.csv. Run "
      "<font name='Mono' size='7.5'>./run_experiments.sh gpu</font> on the GPU machine and rebuild the PDF.]", PENDING)
    rows = [["Graph"] + [GPU_LABEL[v] for v in GPU_VERS]]
    for g in SPARSE:
        rows.append([g] + ["pending"] * 5)
    table("GPU throughput in moves per second (to be filled)", rows, [0.95 * inch] + [0.5 * inch] * 5, key="gpu")

def quality_sentence():
    hit, close = [], []
    for g in GSET:
        if g not in quality or g == "G55":
            continue
        best = max(int(quality[g][v]["best_cut"]) for v in ("cpu_full_omp", "cpu_tiled_bfs_omp"))
        gap = 100 * (BEST_KNOWN[g] - best) / BEST_KNOWN[g]
        (hit if gap <= 0 else close).append((g, gap))
    out = ""
    if hit:
        out += "We reach the best known value on " + " and ".join(g for g, _ in hit) + "."
    if close:
        out += f" On {', '.join(g for g, _ in close)} we are within {max(x for _, x in close):.2f}%."
    return out


subsection("Solution Quality")
qrows = [["Graph", "Best known [4]", "Full (OMP)", "DATS (OMP)", "Gap"]]
for g in GSET:
    if g not in quality:
        continue
    a = int(quality[g]["cpu_full_omp"]["best_cut"])
    b = int(quality[g]["cpu_tiled_bfs_omp"]["best_cut"])
    best = max(a, b)
    gap = 100 * (BEST_KNOWN[g] - best) / BEST_KNOWN[g]
    qrows.append([g, f"{BEST_KNOWN[g]:,}", f"{a:,}", f"{b:,}", f"{gap:.2f}%"])
qt = table(f"Best cut on G-set ({Q_SEARCHES} searches × {Q_STEPS:,} steps)", qrows,
           [0.5 * inch, 0.8 * inch, 0.7 * inch, 0.7 * inch, 0.5 * inch], key="quality")
P(f"Since DATS does not change the search itself, its quality is the same as our BLS-style tabu search. Table {qt} "
  f"compares the best cut of {Q_SEARCHES} searches of {Q_STEPS:,} steps with the best known values reported in [4]. "
  + quality_sentence() + " On the larger G55 the gap is bigger, since BLS uses much longer runs and some extra rules (directed perturbation) that we did not "
  "implement. The full and DATS versions differ slightly only because the BFS relabelling changes the starting "
  "solutions.")
if CONV:
    def conv_cmp(g):
        smax = CONV_STEPS[-1]
        t_tiled = f(CONV[f"{g}-s{smax}"]["cpu_tiled_bfs_omp"], "time_s")
        cut_tiled = int(CONV[f"{g}-s{smax}"]["cpu_tiled_bfs_omp"]["best_cut"])
        ok = [s_ for s_ in CONV_STEPS if f(CONV[f"{g}-s{s_}"]["cpu_full_omp"], "time_s") <= t_tiled]
        s_full = max(ok) if ok else CONV_STEPS[0]
        return t_tiled, cut_tiled, s_full, int(CONV[f"{g}-s{s_full}"]["cpu_full_omp"]["best_cut"])

    conv_graphs = [g for g in ["G22", "G55", "email-Enron"] if f"{g}-s{CONV_STEPS[0]}" in CONV]
    parts_ = []
    for g in conv_graphs:
        t_, c_, s_, cf_ = conv_cmp(g)
        parts_.append(f"on {g}, in {t_:.2f} s the tiled version finishes {CONV_STEPS[-1]:,} steps and finds "
                      f"{c_:,}, while in the same time the full scan only gets through {s_:,} steps and finds {cf_:,}")
    P("Table [[tab:quality]] compares the two versions after the same number of steps. In practice, what matters is "
      "the quality after the same <i>time</i>. Fig. [[fig:convergence]] shows the best cut against wall time for "
      f"runs of {CONV_STEPS[0]:,} to {CONV_STEPS[-1]:,} steps (10 searches, OpenMP). Since the tiled version does "
      "each step faster, its curve is shifted to the left: " + "; ".join(parts_) + ". For a fixed time budget, a "
      "faster step directly gives a better cut.")
    figure("convergence", "Best cut against wall time for runs of 1,000 to 100,000 steps (10 searches with "
           "OpenMP). The dotted line is the best known value from [4].")

section("Discussion")
subsection("Limitations")
P("Our results show that for local search on sparse power-law graphs the cost of choosing the move matters more "
  "than load balancing of the update, and that tiling and degree-aware updates work best together. There are some "
  "limitations. (1) On dense graphs a single flip makes most tiles dirty, so tiling does not help, and on small "
  "graphs the GPU full scan is already so cheap that tiling makes it slower. A simple rule based on <i>n</i> and "
  "the average <i>D</i>(<i>v</i>) could choose between the two methods. (2) The top level still scans "
  "<i>n</i>/32 summaries, which is now the largest part of the work on our biggest graphs. A second level of "
  "summaries (tiles of tiles) would make it about <i>n</i>/1024. (3) The tile summaries must fit in shared "
  "memory for the best speed, which limits <i>n</i> to about 10<super>5</super> on current GPUs. (4) We use one "
  "search per block; for small graphs a warp per search could use the GPU better. (5) Hubs are flipped in very few "
  "steps, so the benefit of the degree-aware update is limited by how often hubs are chosen, which depends on the "
  "search rules. A different search, for example one that flips groups of vertices, could make the update more "
  "important.")
subsection("Threats to Validity")
P("<b>Hardware.</b> The CPU and GPU results come from two different machines (an Apple M5 CPU and a data-centre "
  "GPU on Colab), so the CPU versus GPU ratios depend on these machines. Comparisons inside one device "
  "are not affected by this. Colab GPUs are shared, and we ran each configuration once, so small differences between "
  "GPU variants (a few percent) may be noise. <b>Graphs.</b> We used six G-set graphs, four real SNAP graphs "
  "and synthetic power-law graphs. "
  "Other graph families, such as road networks with very low degree or very dense graphs, may behave differently, "
  "although the cost model of Eq. (5) predicts the trend. <b>Search.</b> We implemented the main BLS rules but not all of "
  "them, and we used fixed parameters on all graphs. Our quality numbers are therefore lower than the best published "
  "ones on hard instances. This does not affect the speed comparisons, because all versions make the same moves. "
  "<b>Metric.</b> Moves per second counts perturbation moves as well, which are cheaper than normal moves on the "
  "GPU. All versions do the same number of perturbation moves, so the comparison stays fair.")
subsection("Which Method Should Be Used?")
P("From the results we can give a simple rule. For small or dense graphs (a few thousand vertices, or "
  "<i>D</i>(<i>v</i>) close to <i>n</i>/32), a full scan with degree-aware updates is the best on the GPU, and the "
  "plain OpenMP full scan is enough on the CPU. For large sparse graphs, especially power-law graphs with hubs, the "
  "full DATS combination is clearly better on both devices. If the vertex ids come in an unknown or random order, "
  "BFS relabelling should be applied, since it is cheap and never hurt much in our tests. Since <i>n</i>, the average "
  "degree and an estimate of <i>D</i>(<i>v</i>) are known before the search starts, this choice could easily be made "
  "automatically by the program.")

# ---------------- VII. conclusion
section("Conclusion")
P("We presented DATS, a GPU tabu search for Max-Cut that combines exact tiled incremental selection, degree-aware "
  "neighbour updates and BFS relabelling. The idea came from noticing that the full scan for the best move, and "
  "not the hub updates, is the main cost of one step on sparse graphs. The tiled selection gives exactly the same "
  f"moves as a full scan while reading {rng_txt(sparse_work, '{:.0f}')}× fewer entries, and with OpenMP on a "
  f"CPU the combined speedup reaches {max(omp_dats_speed):.0f}×. "
  + (f"On a {GPU_NAME} GPU, DATS is {rng_txt([gspeed(g, 'gpu_dats') for g in LARGE])}× faster than a "
     "one-thread-per-vertex design on large sparse graphs, while on small dense graphs a plain full scan remains "
     "better. " if LARGE else "") +
  "We also showed that the best method depends on the graph: for small dense graphs a full scan is better, while for "
  "large sparse graphs the tiled selection wins by a large factor.")
P("In future we want to (1) add a second level of summaries so that the remaining <i>n</i>/32 scan becomes about "
  "<i>n</i>/1024, (2) choose between full scan and tiling automatically from the graph, (3) run several searches per "
  "block, one per warp, for small graphs, (4) try better orderings such as Rabbit Order [23], and (5) apply the "
  "same tiling to other problems that are solved with flip-based local search, such as general QUBO, graph "
  "partitioning and MAX-SAT.")

P("A<font size='7.6'>PPENDIX</font>: R<font size='7.6'>EPRODUCING THE</font> R<font size='7.6'>ESULTS</font>", H1)
P("All code, scripts and data needed to reproduce this paper are in one folder. The graphs are downloaded "
  "automatically from the G-set and SNAP websites. The steps are:", BODY_NI)
cmds = """# CPU experiments (any Linux or macOS machine)
./run_experiments.sh cpu
./run_experiments.sh extra
# GPU experiments (CUDA machine, e.g. Colab T4)
./run_experiments.sh gpu
# build this paper from results/*.csv
pip install reportlab==4.0.9 matplotlib numpy
python3 paper/make_paper.py"""
story.append(Preformatted(cmds, CODE))
story.append(Spacer(1, 4))
P("The single source file <font name='Mono' size='7.5'>maxcut_research.cpp</font> builds with "
  "<font name='Mono' size='7.5'>nvcc -O3 -arch=sm_75 -x cu -Xcompiler -fopenmp</font> for the GPU, or with "
  "<font name='Mono' size='7.5'>g++ -O3 -fopenmp</font> for the CPU only. Table [[tab:params]] lists all search "
  "parameters. Every run prints whether the tiled and full versions, and all GPU variants, gave the same results.")
table("Search parameters used in all experiments", [
    ["Parameter", "Value"],
    ["steps per search", f"{STEPS:,} (quality: {Q_STEPS:,})"],
    ["CPU searches", f"{SEARCHES} (quality and scaling: 10)"],
    ["GPU blocks / threads", f"{gpu[gpu_graphs[0]]['gpu_dats']['searches'] if gpu_graphs else '2 per SM'} / 256"],
    ["tabu tenure", "3 + rand(0, <i>n</i>/10)"],
    ["stall limit before perturbation", "500 steps"],
    ["perturbation strength", "max(2, <i>n</i>/100) to <i>n</i>/20 (capped 100, 500)"],
    ["degree classes", "\u2264 32, 33\u20131024, > 1024"],
    ["tile size", "32 (CPU ablation: 8\u2013128)"],
    ["random generator / seed", "xorshift32 / 2024"],
], [1.55 * inch, 1.85 * inch], key="params")

P("<b>Acknowledgment</b>", ParagraphStyle("ackh", parent=H1, fontName="Stix"))
P("We thank our Parallel Computing course faculty at VJTI for their guidance during this project.")

# ---------------- references
P("R<font size='7.6'>EFERENCES</font>", H1)
refs = [
    "R. M. Karp, “Reducibility among combinatorial problems,” in <i>Complexity of Computer Computations</i>, "
    "R. E. Miller and J. W. Thatcher, Eds. New York: Plenum, 1972, pp. 85–103.",
    "A. Lucas, “Ising formulations of many NP problems,” <i>Frontiers in Physics</i>, vol. 2, art. 5, 2014.",
    "M. X. Goemans and D. P. Williamson, “Improved approximation algorithms for maximum cut and satisfiability "
    "problems using semidefinite programming,” <i>J. ACM</i>, vol. 42, no. 6, pp. 1115–1145, 1995.",
    "U. Benlic and J.-K. Hao, “Breakout local search for the max-cut problem,” <i>Engineering Applications of "
    "Artificial Intelligence</i>, vol. 26, no. 3, pp. 1162–1173, 2013.",
    "C. Cook, H. Zhao, T. Sato, M. Hiromoto, and S. X.-D. Tan, “GPU-based Ising computing for solving max-cut "
    "combinatorial optimization problems,” <i>Integration, the VLSI Journal</i>, Elsevier, 2019.",
    "R. Yasudo, K. Nakano, Y. Ito, M. Tatekawa, R. Katsuki, T. Yazane, and Y. Inaba, “Adaptive bulk search: "
    "Solving quadratic unconstrained binary optimization problems on multiple GPUs,” in <i>Proc. 49th Int. Conf. "
    "Parallel Processing (ICPP)</i>, ACM, 2020.",
    "C. Helmberg and F. Rendl, “A spectral bundle method for semidefinite programming,” <i>SIAM J. "
    "Optimization</i>, vol. 10, no. 3, pp. 673–696, 2000.",
    "F. Glover, “Tabu search—Part I,” <i>ORSA J. Computing</i>, vol. 1, no. 3, pp. 190–206, 1989.",
    "R. Yasudo, K. Nakano, Y. Ito et al., “Graph-theoretic formulation of QUBO for scalable local search on "
    "GPUs,” in <i>Proc. IEEE Int. Parallel and Distributed Processing Symp. Workshops (IPDPSW)</i>, 2022, "
    "pp. 425–434.",
    "D. Merrill, M. Garland, and A. Grimshaw, “Scalable GPU graph traversal,” in <i>Proc. 17th ACM SIGPLAN "
    "Symp. Principles and Practice of Parallel Programming (PPoPP)</i>, 2012, pp. 117–128.",
    "Y. Wang, A. Davidson, Y. Pan, Y. Wu, A. Riffel, and J. D. Owens, “Gunrock: A high-performance graph "
    "processing library on the GPU,” in <i>Proc. 21st ACM SIGPLAN Symp. PPoPP</i>, 2016.",
    "C. M. Fiduccia and R. M. Mattheyses, “A linear-time heuristic for improving network partitions,” in "
    "<i>Proc. 19th Design Automation Conf.</i>, 1982, pp. 175–181.",
    "E. Cuthill and J. McKee, “Reducing the bandwidth of sparse symmetric matrices,” in <i>Proc. 24th ACM "
    "National Conf.</i>, 1969, pp. 157–172.",
    "J. Leskovec and A. Krevl, “SNAP Datasets: Stanford large network dataset collection,” 2014. [Online]. "
    "Available: http://snap.stanford.edu/data",
    "F. Chung and L. Lu, “The average distances in random graphs with given expected degrees,” <i>Proc. "
    "National Academy of Sciences</i>, vol. 99, no. 25, pp. 15879–15882, 2002.",
    "S. Burer, R. D. C. Monteiro, and Y. Zhang, “Rank-two relaxation heuristics for MAX-CUT and other binary "
    "quadratic programs,” <i>SIAM J. Optimization</i>, vol. 12, no. 2, pp. 503–521, 2002.",
    "G. A. Kochenberger, J.-K. Hao, Z. Lü, H. Wang, and F. Glover, “Solving large scale max cut problems via tabu "
    "search,” <i>J. Heuristics</i>, vol. 19, no. 4, pp. 565–571, 2013.",
    "T. Inagaki et al., “A coherent Ising machine for 2000-node optimization problems,” <i>Science</i>, vol. 354, "
    "no. 6312, pp. 603–606, 2016.",
    "H. Goto, K. Tatsumura, and A. R. Dixon, “Combinatorial optimization by simulating adiabatic bifurcations in "
    "nonlinear Hamiltonian systems,” <i>Science Advances</i>, vol. 5, no. 4, eaav2372, 2019.",
    "M. Yamaoka et al., “A 20k-spin Ising chip to solve combinatorial optimization problems with CMOS annealing,” "
    "<i>IEEE J. Solid-State Circuits</i>, vol. 51, no. 1, pp. 303–309, 2016.",
    "S. Hong, S. K. Kim, T. Oguntebi, and K. Olukotun, “Accelerating CUDA graph algorithms at maximum warp,” in "
    "<i>Proc. 16th ACM Symp. PPoPP</i>, 2011, pp. 267–276.",
    "H. Wei, J. X. Yu, C. Lu, and X. Lin, “Speedup graph processing by graph ordering,” in <i>Proc. ACM SIGMOD "
    "Int. Conf. Management of Data</i>, 2016, pp. 1813–1828.",
    "J. Arai, H. Shiokawa, T. Yamamuro, M. Onizuka, and S. Iwamura, “Rabbit Order: Just-in-time parallel "
    "reordering for fast graph analysis,” in <i>Proc. IEEE IPDPS</i>, 2016, pp. 22–31.",
    "A.-L. Barabási and R. Albert, “Emergence of scaling in random networks,” <i>Science</i>, vol. 286, no. 5439, "
    "pp. 509–512, 1999.",
    "NVIDIA, “CUDA C++ Programming Guide.” [Online]. Available: https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
    "Q. Wu and J.-K. Hao, “A memetic approach for the max-cut problem,” in <i>Parallel Problem Solving from Nature "
    "(PPSN XII)</i>, LNCS 7492, Springer, 2012, pp. 297–306.",
    "L. Dagum and R. Menon, “OpenMP: An industry standard API for shared-memory programming,” <i>IEEE "
    "Computational Science and Engineering</i>, vol. 5, no. 1, pp. 46–55, 1998.",
    "G. Marsaglia, “Xorshift RNGs,” <i>J. Statistical Software</i>, vol. 8, no. 14, pp. 1–6, 2003.",
    "M. Harris, “Optimizing parallel reduction in CUDA,” NVIDIA Developer Technology, 2007.",
]
for i, r in enumerate(refs, 1):
    P(f"[{i}]&nbsp;&nbsp;{r}", REF)


# ---------------------------------------------------------------- page layout (IEEE like, 2 columns)
PAGE_W, PAGE_H = letter
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM, GAP = 0.65 * inch, 0.7 * inch, 0.8 * inch, 0.25 * inch
COL_W = (PAGE_W - 2 * MARGIN_X - GAP) / 2
TITLE_H = 1.75 * inch


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Stix", 8)
    canvas.drawCentredString(PAGE_W / 2, 0.45 * inch, str(doc.page))
    canvas.restoreState()


doc = BaseDocTemplate(OUT_PDF, pagesize=letter, title="DATS: Degree-Aware Tiled Search for Fast Max-Cut Local Search on GPUs",
                      author="Vedant Tamhanekar, Siddhant Salunke", leftMargin=MARGIN_X, rightMargin=MARGIN_X,
                      topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM)
body_h = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM
first = [
    Frame(MARGIN_X, PAGE_H - MARGIN_TOP - TITLE_H, PAGE_W - 2 * MARGIN_X, TITLE_H, id="title", leftPadding=0, rightPadding=0),
    Frame(MARGIN_X, MARGIN_BOTTOM, COL_W, body_h - TITLE_H, id="c1", leftPadding=0, rightPadding=0, topPadding=2),
    Frame(MARGIN_X + COL_W + GAP, MARGIN_BOTTOM, COL_W, body_h - TITLE_H, id="c2", leftPadding=0, rightPadding=0, topPadding=2),
]
later = [
    Frame(MARGIN_X, MARGIN_BOTTOM, COL_W, body_h, id="l1", leftPadding=0, rightPadding=0),
    Frame(MARGIN_X + COL_W + GAP, MARGIN_BOTTOM, COL_W, body_h, id="l2", leftPadding=0, rightPadding=0),
]
doc.addPageTemplates([PageTemplate(id="First", frames=first, onPage=footer),
                      PageTemplate(id="Later", frames=later, onPage=footer)])
story.insert(0, NextPageTemplate("Later"))
doc.build(story)
print("wrote", OUT_PDF)
