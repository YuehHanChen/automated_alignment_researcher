"""Shared style for Bruce's paper figures. Single source of truth.

Anything an agent needs to follow conventions reproducibly lives here:

  • Width invariants: HALF_PAGE_W, FULL_PAGE_W (paper widths in inches).
  • Spacing invariants: gaps between legend / suptitle / rows / columns.
    Same numbers across ALL templates so figures look like a family.
  • figsize_for(): compute (W, H) given a width and a row count.
  • apply_layout():  apply standard subplots_adjust + place suptitle + legend
                     with correct inch-anchored positions. Templates call this
                     ONCE; they don't pick layout numbers themselves.
  • setup_rcparams(), palette(), better_arrow(), save_figure(): unchanged.

Why centralised? Per-template magic numbers were producing inconsistent
spacing across our figures. The values below are the ONE place those
numbers should ever live.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import rcParams

try:
    import cmcrameri.cm as cmc
    _BATLOW = cmc.batlow
except ImportError:
    _BATLOW = None


# =============================================================================
# WIDTH INVARIANTS (paper-derived, do not change)
# =============================================================================
HALF_PAGE_W = 3.25      # one paper column (NeurIPS / ICML / ACL)
FULL_PAGE_W = 6.75      # full text width


# =============================================================================
# SPACING INVARIANTS (inches) — DO NOT vary per template
# =============================================================================
# Vertical anchors, measured from the top of the figure (inches).
# Layout stack (top to bottom):
#   [SUPTITLE_FROM_TOP]      ─ suptitle (optional, ~0.20" tall)
#   [LEGEND_FROM_TOP*]       ─ legend (~0.25" tall for one row)
#   [ROW1_FROM_TOP*]         ─ top of first panel row
#   row 1 panels (PANEL_H tall)
#   [ROW_TO_ROW gap]
#   row 2 panels
#   ...
#   [BOTTOM_PAD]             ─ space for bottom x-label
SUPTITLE_FROM_TOP   = 0.18   # top of suptitle text
LEGEND_FROM_TOP     = 0.20   # top of legend (no suptitle)
LEGEND_FROM_TOP_SUP = 0.55   # top of legend (suptitle present)
ROW1_FROM_TOP       = 0.70   # top of row 1 (no suptitle) — leaves ~0.25" after legend
ROW1_FROM_TOP_SUP   = 1.05   # top of row 1 (suptitle present) — leaves ~0.25" after legend

# Inter-panel gaps. Row gap accommodates: bottom row's x-label tick numbers,
# the row's xlabel, and the next row's panel title.
ROW_TO_ROW = 0.70
# Column gap depends on whether y-axis is shared:
#  - SHARED:    only the leftmost panel has y-ticks + label, so a narrow gap is fine.
#  - INDIV:     every panel has its own y-tick numbers + y-label that need to fit in the gap.
COL_TO_COL_SHARED = 0.32
COL_TO_COL_INDIV  = 0.75
# Backward-compatible alias (defaults to shared)
COL_TO_COL = COL_TO_COL_SHARED

# Bottom margin (room for x-axis ticks + xlabel)
BOTTOM_PAD = 0.60

# Default per-row panel height (used by figsize_for)
PANEL_H_DEFAULT = 1.7

# Fixed utility colors
GRID = "#e5e7eb"
REFERENCE = "#333333"
POSITIVE = "#2ca02c"
NEGATIVE = "#d62728"
NEUTRAL = "#888888"


# =============================================================================
# RCPARAMS
# =============================================================================
def setup_rcparams() -> None:
    """Apply Bruce's typography + spine conventions globally."""
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,        # subfigure title (a) (b) (c)
        "axes.labelsize": 8,        # x/y axis labels
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.titlesize": 11,        # suptitle
        "figure.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "figure.dpi": 150,
    })


# =============================================================================
# PALETTE
# =============================================================================
_HEX_FALLBACK = {
    "baseline":   "#0f3c5f",
    "control":    "#30685c",
    "primary":    "#6c7c3c",
    "secondary":  "#99882c",
    "tertiary":   "#b38e2f",
    "highlight":  "#f29d6d",
    "extreme":    "#fdafa7",
}
_POSITIONS = {
    "baseline":  0.10,
    "control":   0.30,
    "primary":   0.45,
    "secondary": 0.55,
    "tertiary":  0.65,
    "highlight": 0.78,
    "extreme":   0.88,
}

def palette() -> dict[str, tuple]:
    """Return a {role: rgba} dict. If cmcrameri is missing, falls back to hex."""
    if _BATLOW is not None:
        return {k: _BATLOW(v) for k, v in _POSITIONS.items()}
    return dict(_HEX_FALLBACK)


# =============================================================================
# LAYOUT — single source of truth for fig sizing + suptitle/legend placement
# =============================================================================
def figsize_for(width: float, *, n_rows: int = 1,
                panel_h: float = PANEL_H_DEFAULT,
                has_suptitle: bool = False) -> tuple[float, float]:
    """Compute (W, H) for a uniform-grid figure.

    width: pick HALF_PAGE_W or FULL_PAGE_W.
    n_rows: number of panel rows.
    panel_h: per-row panel height (inches).
    has_suptitle: reserve extra top space for centered suptitle.
    """
    top = ROW1_FROM_TOP_SUP if has_suptitle else ROW1_FROM_TOP
    h = top + n_rows * panel_h + (n_rows - 1) * ROW_TO_ROW + BOTTOM_PAD
    return (width, h)


def apply_layout(fig, fig_h: float, *, n_rows: int = 1, n_cols: int = 1,
                 panel_h: float = PANEL_H_DEFAULT,
                 panel_w: float | None = None,
                 share_y: bool = True,
                 has_suptitle: bool = False,
                 suptitle: str | None = None,
                 legend_handles: Iterable | None = None,
                 legend_ncol: int | None = None) -> dict:
    """Configure a figure's standard layout in one call.

    Sets subplots_adjust(top, bottom, wspace, hspace) using the SHARED
    inch-based spacing constants. Optionally centers a suptitle and
    places a figure-level legend at the canonical inch offset from top.

    share_y: True if all panels share the y-axis (default — narrower gap).
             False if each panel has its own y-label and tick numbers
             (wider gap to fit them).

    Returns a dict containing the legend object (if created).
    """
    has_sup = has_suptitle or (suptitle is not None)

    top_inches = ROW1_FROM_TOP_SUP if has_sup else ROW1_FROM_TOP
    adjust = dict(
        top=1 - top_inches / fig_h,
        bottom=BOTTOM_PAD / fig_h,
    )
    col_gap = COL_TO_COL_SHARED if share_y else COL_TO_COL_INDIV
    if n_cols > 1:
        # Approximate panel width if not provided. Matplotlib default
        # left=0.125, right=0.9 → usable horizontal frac = 0.775.
        if panel_w is None:
            panel_w = (fig.get_figwidth() * 0.775) / n_cols
        adjust["wspace"] = col_gap / panel_w
    if n_rows > 1:
        adjust["hspace"] = ROW_TO_ROW / panel_h
    fig.subplots_adjust(**adjust)

    if suptitle is not None:
        sup_y = 1 - SUPTITLE_FROM_TOP / fig_h
        fig.suptitle(suptitle, y=sup_y, va="top")  # rcParams handles size/weight

    legend_obj = None
    if legend_handles is not None:
        leg_top_inches = LEGEND_FROM_TOP_SUP if has_sup else LEGEND_FROM_TOP
        leg_y = 1 - leg_top_inches / fig_h
        legend_obj = fig.legend(
            handles=list(legend_handles),
            loc="upper center", bbox_to_anchor=(0.5, leg_y),
            ncol=legend_ncol or len(list(legend_handles)),
            frameon=True, fancybox=True, framealpha=0.95,
            handletextpad=0.4, columnspacing=1.5, borderpad=0.4,
        )

    return {"legend": legend_obj}


# =============================================================================
# HELPERS
# =============================================================================
# Better-arrow geometry — FIXED in inches. Don't shrink to avoid overlap;
# move the arrow to a different corner instead.
ARROW_PAD_INCHES       = 0.07
ARROW_LEN_INCHES       = 0.22
ARROW_LABEL_GAP_INCHES = 0.05


def better_arrow(ax, *, direction: str = "up", label: str = "Better",
                 corner: str = "lower right") -> None:
    """Draw a 'Better' arrow inside an axes corner with FIXED inch dimensions.

    Call this AFTER apply_layout() so ax.get_position() reflects the final
    panel size. Arrow length, label gap, and corner pad are constant inches
    regardless of panel size — this guarantees that arrows look identical
    across templates of different widths.

    direction: 'up', 'down', 'left', 'right'.
    corner:    'lower right', 'upper right', 'lower left', 'upper left'.

    If the arrow overlaps data, MOVE the corner. Never shrink the arrow.
    """
    fig = ax.figure
    bbox = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    panel_w = bbox.width * fig_w
    panel_h = bbox.height * fig_h

    pad_x = ARROW_PAD_INCHES / panel_w
    pad_y = ARROW_PAD_INCHES / panel_h
    span_y = ARROW_LEN_INCHES / panel_h
    span_x = ARROW_LEN_INCHES / panel_w
    label_gap_x = ARROW_LABEL_GAP_INCHES / panel_w
    label_gap_y = ARROW_LABEL_GAP_INCHES / panel_h

    if "right" in corner:
        x = 1 - pad_x
        ha = "right"
    else:
        x = pad_x
        ha = "left"
    if "lower" in corner:
        y_base, y_tip = pad_y, pad_y + span_y
    else:
        y_base, y_tip = 1 - pad_y - span_y, 1 - pad_y

    if direction == "down":
        y_base, y_tip = y_tip, y_base
    elif direction in ("left", "right"):
        x_base, x_tip = (x + span_x, x) if direction == "left" else (x - span_x, x)
        y_base = y_tip = (y_base + y_tip) / 2
        ax.annotate("", xy=(x_tip, y_tip), xytext=(x_base, y_base),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    lw=1.0, mutation_scale=8))
        label_x = (x_base + x_tip) / 2
        ax.text(label_x, y_tip + label_gap_y, label,
                transform=ax.transAxes,
                fontsize=8, fontstyle="italic", ha="center", va="bottom")
        return

    ax.annotate("", xy=(x, y_tip), xytext=(x, y_base),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=1.0, mutation_scale=8))
    ax.text(x - (label_gap_x if ha == "right" else -label_gap_x),
            (y_base + y_tip) / 2, label,
            transform=ax.transAxes,
            fontsize=8, fontstyle="italic", ha=ha, va="center")


def save_figure(fig, name: str, out_dir: str | Path = "figures") -> Path:
    """Save PNG (600 DPI) + PDF (vector) with tight bbox."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{name}.png"
    pdf = out / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    return png
