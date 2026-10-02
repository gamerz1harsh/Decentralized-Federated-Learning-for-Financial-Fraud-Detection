"""
Generate flowchart figures for the project report.
Clean, professional flowcharts styled similar to draw.io (boxes + arrows).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def draw_box(ax, x, y, w, h, text, fc="#EAF2FA", ec="#2E75B6", fs=9, bold=False, rounded=True):
    """Draw a rounded rectangle box with centered text."""
    boxstyle = "round,pad=0.02,rounding_size=0.02" if rounded else "square,pad=0.02"
    patch = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=boxstyle,
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=fs, fontweight=weight, zorder=3,
        wrap=True,
    )


def draw_arrow(ax, x1, y1, x2, y2, color="#2E75B6", style="-|>", lw=1.6, ls="-"):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        linestyle=ls,
        zorder=1,
    )
    ax.add_patch(arrow)


def draw_arrow_curve(ax, x1, y1, x2, y2, color="#2E75B6", lw=1.6, connectionstyle="arc3,rad=0.15"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        zorder=1,
    )
    ax.add_patch(arrow)


def setup_ax(fig, xlim=(0, 10), ylim=(0, 10)):
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return ax


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------
# Figure 1: System Architecture
# ---------------------------------------------------------------

fig = plt.figure(figsize=(10, 6.5))
ax = setup_ax(fig, xlim=(0, 10), ylim=(0, 10))

# Title handled in docx caption

# Banks (clients) on left
draw_box(ax, 1.3, 8.2, 2.2, 1.0, "Bank A\n(local data)", fc="#E2EFDA", ec="#538135", fs=9)
draw_box(ax, 1.3, 6.2, 2.2, 1.0, "Bank B\n(local data)", fc="#E2EFDA", ec="#538135", fs=9)
draw_box(ax, 1.3, 4.2, 2.2, 1.0, "Bank C\n(local data)", fc="#E2EFDA", ec="#538135", fs=9)
draw_box(ax, 1.3, 2.2, 2.2, 1.0, "Bank D\n(local data)", fc="#E2EFDA", ec="#538135", fs=9)

# Central server box
draw_box(ax, 5.0, 5.2, 2.6, 5.8, "", fc="#FCE4D6", ec="#C55A11", fs=9)
ax.text(5.0, 8.0, "Central Server", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#C55A11")
ax.text(5.0, 7.5, "Model Aggregation\n& Client Scoring", ha="center", va="center",
        fontsize=9, color="#C55A11")

# Scoring module inside server
draw_box(ax, 5.0, 6.0, 2.0, 1.6, "Multi-Dimensional\nScoring\n(Trust, Quality,\nNovelty, Complementarity,\nTemporal)", fc="#FFF2CC", ec="#BF9000", fs=7.5)
# Aggregation module
draw_box(ax, 5.0, 3.4, 2.0, 1.4, "Weighted\nAggregation\n(Contribution-Aware)", fc="#DEEBF7", ec="#2E75B6", fs=8)

# Arrows: banks -> server (upload weights)
for by in (8.2, 6.2, 4.2, 2.2):
    draw_arrow(ax, 2.4, by, 3.7, 6.0 if by >= 4.2 else 3.4, color="#538135")

# Server -> banks (distribute global model)
for by in (8.2, 6.2, 4.2, 2.2):
    draw_arrow_curve(ax, 6.3, 6.0 if by >= 4.2 else 3.4, 7.7, by, color="#C55A11", connectionstyle="arc3,rad=-0.2")

# Labels on arrows
ax.text(3.0, 8.6, "Upload", fontsize=7, color="#538135", style="italic")
ax.text(3.05, 6.6, "local model", fontsize=7, color="#538135", style="italic")
ax.text(3.05, 4.55, "weights", fontsize=7, color="#538135", style="italic")
ax.text(3.05, 1.35, "(", fontsize=7, color="#538135", style="italic")
ax.text(7.0, 8.6, "Distribute global model", fontsize=7, color="#C55A11", style="italic")

# Global model output
draw_box(ax, 8.6, 5.2, 2.0, 1.2, "Global\nFraud Detection\nModel", fc="#DDEBF7", ec="#2E75B6", fs=9, bold=True)
draw_arrow(ax, 6.3, 5.2, 7.6, 5.2, color="#C55A11")

# Legend-ish label
ax.text(5.0, 1.0, "Each bank trains privately on its own data; only weights are shared.",
        ha="center", fontsize=8, style="italic", color="#555555")

save_fig(fig, "fig1_system_architecture.png")


# ---------------------------------------------------------------
# Figure 2: Federated Learning Round Workflow
# ---------------------------------------------------------------

fig = plt.figure(figsize=(10, 6))
ax = setup_ax(fig, xlim=(0, 10), ylim=(0, 10))

# Top: server distributes
draw_box(ax, 5.0, 9.2, 4.5, 1.1, "Server distributes current global model", fc="#DEEBF7", ec="#2E75B6", fs=10, bold=True)

# Clients row
draw_box(ax, 1.0, 7.2, 1.9, 1.0, "Client 1\nLocal Training", fc="#E2EFDA", ec="#538135", fs=8)
draw_box(ax, 3.7, 7.2, 1.9, 1.0, "Client 2\nLocal Training", fc="#E2EFDA", ec="#538135", fs=8)
draw_box(ax, 6.4, 7.2, 1.9, 1.0, "Client 3\nLocal Training", fc="#E2EFDA", ec="#538135", fs=8)
draw_box(ax, 9.0, 7.2, 1.9, 1.0, "Client N\nLocal Training", fc="#E2EFDA", ec="#538135", fs=8)

for cx in (1.0, 3.7, 6.4, 9.0):
    draw_arrow(ax, cx, 8.6, cx, 8.0, color="#2E75B6")

# Clients send weights up
draw_box(ax, 5.0, 5.4, 4.8, 1.1, "Clients upload model weights to server", fc="#FFF2CC", ec="#BF9000", fs=9)

for cx in (1.0, 3.7, 6.4, 9.0):
    draw_arrow(ax, cx, 6.7, cx, 6.1, color="#538135")

# Scoring box
draw_box(ax, 5.0, 3.9, 5.6, 1.1, "Server scores each client's contribution\n(Quality, Trust, Novelty, Complementarity, Temporal)", fc="#FCE4D6", ec="#C55A11", fs=8)
draw_arrow(ax, 5.0, 4.8, 5.0, 4.5, color="#BF9000")

# Aggregation
draw_box(ax, 5.0, 2.5, 4.6, 1.1, "Contribution-aware weighted aggregation", fc="#DEEBF7", ec="#2E75B6", fs=9, bold=True)
draw_arrow(ax, 5.0, 3.3, 5.0, 3.1, color="#C55A11")

# Updated global model
draw_box(ax, 5.0, 1.1, 4.6, 1.0, "Updated global model\n→ next round", fc="#E2EFDA", ec="#538135", fs=9)
draw_arrow(ax, 5.0, 1.9, 5.0, 1.7, color="#2E75B6")

# Loop back arrow (right side)
draw_arrow_curve(ax, 7.3, 1.1, 7.3, 8.6, color="#C00000", lw=2.0, connectionstyle="arc3,rad=-0.25")
ax.text(7.5, 4.8, "Repeat for R rounds", fontsize=8, rotation=90, color="#C00000", style="italic")

save_fig(fig, "fig2_training_workflow.png")


# ---------------------------------------------------------------
# Figure 3: Multi-Dimensional Scoring Pipeline
# ---------------------------------------------------------------

fig = plt.figure(figsize=(10, 6.5))
ax = setup_ax(fig, xlim=(0, 10), ylim=(0, 10))

# Input
draw_box(ax, 5.0, 9.0, 5.2, 1.2, "Client model weights submitted each round", fc="#DEEBF7", ec="#2E75B6", fs=10, bold=True)

# Five dimension boxes
dims = [
    (1.0, 6.8, "Quality\n(local performance,\nloss, accuracy)", "#E2EFDA", "#538135"),
    (3.2, 6.8, "Trust\n(consistency,\nreliability over rounds)", "#E2EFDA", "#538135"),
    (5.4, 6.8, "Novelty\n(deviation from\ncurrent global model)", "#E2EFDA", "#538135"),
    (7.6, 6.8, "Complementarity\n(cross-client\nutility of update)", "#E2EFDA", "#538135"),
    (9.8, 6.8, "Temporal Trend\n(improving vs.\ndeteriorating)", "#E2EFDA", "#538135"),
]

for dx, dy, text, fc, ec in dims:
    draw_box(ax, dx, dy, 1.7, 1.8, text, fc=fc, ec=ec, fs=7)

for dx, dy, text, fc, ec in dims:
    draw_arrow(ax, dx, 8.4, dx, 7.8, color="#2E75B6")

# Normalization / combination
draw_box(ax, 5.0, 4.4, 6.4, 1.4, "Score normalization & combination\n(weighted or learned fusion)", fc="#FFF2CC", ec="#BF9000", fs=8.5)

for dx, dy, text, fc, ec in dims:
    draw_arrow(ax, dx, 5.8, dx, 5.2, color="#538135")

# Aggregation weight
draw_box(ax, 5.0, 2.6, 5.6, 1.2, "Per-client aggregation weight\n(softmax over combined scores)", fc="#FCE4D6", ec="#C55A11", fs=9, bold=True)
draw_arrow(ax, 5.0, 3.7, 5.0, 3.3, color="#BF9000")

# Global aggregation
draw_box(ax, 5.0, 1.0, 4.6, 1.0, "Contribution-weighted\nFedAvg-style aggregation", fc="#DEEBF7", ec="#2E75B6", fs=9)
draw_arrow(ax, 5.0, 2.0, 5.0, 1.6, color="#C55A11")

save_fig(fig, "fig3_scoring_pipeline.png")


# ---------------------------------------------------------------
# Figure 4: Data Partitioning (Non-IID)
# ---------------------------------------------------------------

fig = plt.figure(figsize=(10, 5.5))
ax = setup_ax(fig, xlim=(0, 10), ylim=(0, 10))

# Raw dataset
draw_box(ax, 5.0, 9.2, 4.8, 1.2, "Credit Card Fraud Dataset\n(284,807 transactions, 492 fraud cases)", fc="#DEEBF7", ec="#2E75B6", fs=9, bold=True)

# Preprocessing
draw_box(ax, 5.0, 7.4, 5.4, 1.2, "Preprocessing\n(standardization, class handling,\nremoval of duplicates, train/test split)", fc="#E2EFDA", ec="#538135", fs=8)
draw_arrow(ax, 5.0, 8.6, 5.0, 8.1, color="#2E75B6")

# Split into train
draw_box(ax, 2.5, 5.6, 3.4, 1.2, "Training Set\n(~80%)", fc="#FFF2CC", ec="#BF9000", fs=9)
draw_box(ax, 7.7, 5.6, 3.2, 1.2, "Test Set\n(~20%)", fc="#FCE4D6", ec="#C55A11", fs=9)
draw_arrow(ax, 3.5, 6.8, 3.5, 6.3, color="#538135")
draw_arrow(ax, 6.7, 6.8, 6.7, 6.3, color="#538135")

# Dirichlet split
draw_box(ax, 2.5, 3.6, 4.4, 1.2, "Dirichlet Partitioning (α = 0.1)\nNon-IID distribution\n(fraud patterns vary per bank)", fc="#DDEBF7", ec="#2E75B6", fs=8)
draw_arrow(ax, 2.5, 5.0, 2.5, 4.3, color="#BF9000")

# Four banks
bank_colors = ["#E2EFDA", "#FCE4D6", "#FFF2CC", "#DEEBF7"]
bank_edges = ["#538135", "#C55A11", "#BF9000", "#2E75B6"]
for i, (bx, by) in enumerate([(0.9, 1.6), (2.9, 1.6), (4.9, 1.6), (6.9, 1.6)]):
    draw_box(ax, bx, by, 1.7, 1.1, f"Bank {chr(65+i)}\nnon-IID shard", fc=bank_colors[i], ec=bank_edges[i], fs=8)
    draw_arrow(ax, 2.5, 3.0, bx, 2.3, color="#2E75B6")

# Test set note
draw_box(ax, 7.7, 3.6, 4.2, 1.2, "Held-out Test Set\n(global evaluation of\naggregated model)", fc="#FCE4D6", ec="#C55A11", fs=8)
draw_arrow(ax, 7.7, 5.0, 7.7, 4.3, color="#C55A11")

save_fig(fig, "fig4_data_partitioning.png")


# ---------------------------------------------------------------
# Figure 5: Overall Methodology
# ---------------------------------------------------------------

fig = plt.figure(figsize=(10, 7.5))
ax = setup_ax(fig, xlim=(0, 10), ylim=(0, 12))

steps = [
    (5.0, 11.2, "Step 1: Data Preparation\n(Dataset, preprocessing, train/test split)", "#DEEBF7", "#2E75B6"),
    (5.0, 9.6, "Step 2: Non-IID Partitioning\n(Dirichlet split into 4 bank shards)", "#DEEBF7", "#2E75B6"),
    (5.0, 8.0, "Step 3: Local Model Training\n(each bank trains an MLP on private data)", "#E2EFDA", "#538135"),
    (5.0, 6.4, "Step 4: Upload Model Weights\nto central server", "#FFF2CC", "#BF9000"),
    (5.0, 4.8, "Step 5: Multi-Dimensional Client Scoring\n(Quality, Trust, Novelty, Complementarity, Temporal)", "#FCE4D6", "#C55A11"),
    (5.0, 3.2, "Step 6: Contribution-Aware Aggregation\n(weighted average using computed scores)", "#DEEBF7", "#2E75B6"),
    (5.0, 1.6, "Step 7: Evaluate & Distribute\n(Evaluate global model, send back to banks)", "#E2EFDA", "#538135"),
]

for i, (sx, sy, text, fc, ec) in enumerate(steps):
    bold = i in (0, 5)
    draw_box(ax, sx, sy, 6.2, 1.1, text, fc=fc, ec=ec, fs=8.5, bold=bold)
    if i < len(steps) - 1:
        draw_arrow(ax, sx, sy - 0.55, steps[i+1][1] if steps[i+1][0] == sx else sx, steps[i+1][1] + 0.55, color="#2E75B6")

# Loop back
draw_arrow_curve(ax, 8.1, 1.6, 8.1, 8.5, color="#C00000", lw=1.8, connectionstyle="arc3,rad=-0.2")
ax.text(8.35, 5.0, "Loop until\nconvergence", fontsize=7.5, rotation=75, color="#C00000", style="italic")

save_fig(fig, "fig5_methodology.png")


# ---------------------------------------------------------------
# Figure 6: FedAvg vs Proposed (concept comparison)
# ---------------------------------------------------------------

fig = plt.figure(figsize=(10, 5))
ax = setup_ax(fig, xlim=(0, 10), ylim=(0, 10))

# FedAvg path
draw_box(ax, 2.5, 8.6, 4.4, 1.3, "Standard FedAvg", fc="#F2F2F2", ec="#7F7F7F", fs=10, bold=True)
draw_box(ax, 2.5, 6.8, 4.6, 1.1, "Weights ∝ number of local samples\n(sample-size weighting only)", fc="#F2F2F2", ec="#7F7F7F", fs=8)
draw_box(ax, 2.5, 4.9, 3.6, 1.1, "Averaged global model", fc="#F2F2F2", ec="#7F7F7F", fs=8.5)
draw_arrow(ax, 2.5, 7.9, 2.5, 7.4, color="#7F7F7F")
draw_arrow(ax, 2.5, 6.2, 2.5, 5.5, color="#7F7F7F")

# Proposed path
draw_box(ax, 7.5, 8.6, 4.4, 1.3, "Proposed Contribution-Aware FL", fc="#DEEBF7", ec="#2E75B6", fs=10, bold=True)
draw_box(ax, 7.5, 6.8, 5.2, 1.1, "Weights ∝ multi-dimensional scores\n(Quality, Trust, Novelty, Complementarity,\nTemporal trend)", fc="#E2EFDA", ec="#538135", fs=7.5)
draw_box(ax, 7.5, 4.9, 4.2, 1.1, "Task-aware aggregated model", fc="#DEEBF7", ec="#2E75B6", fs=8.5)
draw_arrow(ax, 7.5, 7.9, 7.5, 7.4, color="#2E75B6")
draw_arrow(ax, 7.5, 6.2, 7.5, 5.5, color="#2E75B6")

# Comparison bracket labels
ax.text(0.4, 6.3, "Less", fontsize=8, color="#7F7F7F", style="italic")
ax.text(10.0, 6.3, "More", fontsize=8, color="#2E75B6", style="italic")
ax.text(5.0, 3.9, "Key difference: the proposed method weighs clients by learned contribution\nrather than by sample count alone.", ha="center", fontsize=8.5, style="italic", color="#404040")

save_fig(fig, "fig6_comparison.png")

print("\nAll figures generated successfully.")