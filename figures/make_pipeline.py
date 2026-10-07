#!/usr/bin/env python3
"""Figure 1: what a best-of-N verifier pipeline actually measures.

The previous version of this figure was a plumbing diagram -- five boxes in a
row -- which restated the abstract and carried none of the paper's claim. This
one shows the object the paper is about: ONE candidate pool, THREE selection
rules over that same pool, and the two differences we separate,

    Delta_maj = selector - majority@8    (does it beat self-consistency?)
    SI        = selector - standalone    (do the candidates add anything?)

Grounding is carried by text rather than by the old dashed arc, which spent
~40% of the canvas restating a fact already printed inside the verifier box.
The input panel says the video's frames are read by both models; the two
verifier lanes each name their own K-frame budget.

Two drawing rules make it legible at publication size:

  1. The canvas is drawn at exactly the width it is displayed at
     (0.98 * 5.5in of the NeurIPS text block), so a fontsize of 6.3 here is
     6.3pt on the page. The previous figure was drawn 10.9in wide and shown at
     4.29in -- a 2.54x downscale that rendered its 8.2pt body text at ~3.2pt.
  2. x and y units are square (ylim = 100 / ASPECT), so rounded corners and
     arrowheads are not sheared.

ASPECT is the page-budget knob. Figure 1 floats to the top of page 2, where the
Introduction ends exactly at the last line -- there is no slack, so the float
must not grow. Height on the page = 0.98 * 5.5 / ASPECT inches; the figure this
replaces was 1.1807in tall, so ASPECT >= 4.56 cannot cost a line.
"""
import os

for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(v, None)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- typography: STIXGeneral is Times-metric-compatible, matching NeurIPS ----
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "text.usetex": False,
})

# --- geometry ---------------------------------------------------------------
DISP_W = 0.98 * 5.5          # inches the float occupies on the page
ASPECT = 4.58                # width / height; >= 4.56 cannot grow the float
FIG_W, FIG_H = DISP_W, DISP_W / ASPECT
YMAX = 100.0 / ASPECT        # square units
PT = 3.88                    # points per unit, at 1:1 scale (100u = 5.39in)

# --- palette ---------------------------------------------------------------
INK = "#15181C"
GRAY = "#5B6470"      # the video/question input
BLUE = "#0064E0"      # base generator
AMBER = "#B26A00"      # candidate pool
SLATE = "#6E7681"      # majority@8 -- the free baseline
RED = "#D6372B"      # the verifier, in both of its roles
MUTED = "#8A9099"      # organising labels above the two halves
NOTE = "#6B7280"      # secondary notes inside panels (dark enough at 4.8pt)

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, YMAX)
ax.axis("off")

TS, SS, NS = 6.3, 5.15, 4.8   # title / subtitle / note point sizes

# The generation row and the lane stack span an identical vertical band, so the
# two halves of the figure read as flush blocks.
FY0, FY1 = 2.80, 19.00
CY = (FY0 + FY1) / 2
LANE_H, LANE_GAP = 4.90, 0.75
LANE_CY = [FY1 - LANE_H / 2, CY, FY0 + LANE_H / 2]


def panel(x0, x1, y0, y1, color, lines, lw=0.9, tint="12", dashed=False, dy=0.0):
    """Rounded box with a vertically centred stack of (text, size, bold, color)."""
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.85",
        linewidth=lw, edgecolor=color, facecolor=color + tint,
        linestyle=(0, (2.6, 1.6)) if dashed else "solid",
        mutation_aspect=1.0, zorder=2))
    heights = [ln[1] * 1.42 / PT for ln in lines]
    cur = (y0 + y1) / 2 + sum(heights) / 2 + dy
    for ln, h in zip(lines, heights):
        txt, size, bold = ln[0], ln[1], ln[2]
        col = ln[3] if len(ln) > 3 else (color if bold else INK)
        cur -= h / 2
        ax.text((x0 + x1) / 2, cur, txt, ha="center", va="center",
                fontsize=size, color=col,
                fontweight="bold" if bold else "normal", zorder=3)
        cur -= h / 2


def arrow(xa, ya, xb, yb, color=INK, lw=0.85, scale=5.6):
    ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>",
                                 mutation_scale=scale, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=4))


# --- left: generate the candidate pool, once -------------------------------
panel(1.2, 15.4, FY0, FY1, GRAY, [
    ("video + question", TS, True),
    ("options A–F", SS, False),
    ("read by both models", NS, False, NOTE),
])
panel(18.4, 35.6, FY0, FY1, BLUE, [
    ("base generator", TS, True),
    ("Qwen3.5-27B", SS, False),
    ("64 frames, $T{=}0.8$", SS, False),
])
panel(38.6, 54.0, FY0, FY1, AMBER, [
    ("candidate pool $\\mathcal{C}$", TS, True),
    ("$N{=}8$ CoT rollouts", SS, False),
    ("one answer letter each", NS, False, NOTE),
], dy=1.2)

arrow(15.4, CY, 18.4, CY)
arrow(35.6, CY, 38.6, CY)

# the pool, drawn: eight sampled answer letters, with repeats
CHIP_W, CHIP_H, CHIP_GAP = 1.58, 1.72, 0.30
letters = ["A", "C", "A", "B", "A", "D", "A", "C"]
row_w = len(letters) * CHIP_W + (len(letters) - 1) * CHIP_GAP
x = (38.6 + 54.0) / 2 - row_w / 2
chip_cy = 7.90                       # tucked just under the panel's text stack
for ch in letters:
    ax.add_patch(FancyBboxPatch(
        (x, chip_cy - CHIP_H / 2), CHIP_W, CHIP_H,
        boxstyle="round,pad=0,rounding_size=0.3",
        linewidth=0.55, edgecolor=AMBER, facecolor="#FFFFFF",
        mutation_aspect=1.0, zorder=3))
    ax.text(x + CHIP_W / 2, chip_cy, ch, ha="center", va="center",
            fontsize=4.95, color=AMBER, zorder=4)
    x += CHIP_W + CHIP_GAP

# --- right: three selection rules over that same pool ----------------------
LX0, LX1, BUS = 58.4, 84.2, 56.2

panel(LX0, LX1, LANE_CY[0] - LANE_H / 2, LANE_CY[0] + LANE_H / 2, SLATE, [
    ("majority@8 over $\\mathcal{C}$", TS, True),
    ("self-consistency; no extra compute", SS, False),
])
panel(LX0, LX1, LANE_CY[1] - LANE_H / 2, LANE_CY[1] + LANE_H / 2, RED, [
    ("verifier selects from $\\mathcal{C}$", TS, True),
    ("2nd VLM re-watches $K$ frames", SS, False),
], lw=1.35, tint="1E")
panel(LX0, LX1, LANE_CY[2] - LANE_H / 2, LANE_CY[2] + LANE_H / 2, RED, [
    ("same verifier, alone", TS, True),
    ("same $K$ frames, $\\mathcal{C}$ withheld", SS, False),
], lw=0.85, tint="09", dashed=True)

# a bus, so the three rules visibly read off one and the same pool
ax.plot([54.0, BUS], [CY, CY], lw=0.85, color=INK, zorder=1)
ax.plot([BUS, BUS], [LANE_CY[0], LANE_CY[2]], lw=0.85, color=INK,
        solid_capstyle="round", zorder=1)
for ly in LANE_CY:
    arrow(BUS, ly, LX0, ly)

# --- far right: the two axes, as differences between lanes ------------------
TICK, BX, LBLX, BRK = 84.6, 86.0, 86.9, 0.45


def axis_bracket(y_hi, y_lo, name, gloss):
    ax.plot([TICK, BX], [y_hi, y_hi], lw=0.75, color=INK, zorder=3)
    ax.plot([TICK, BX], [y_lo, y_lo], lw=0.75, color=INK, zorder=3)
    # break the vertical at the shared middle lane so the two brackets read as two
    lo = y_lo + BRK if y_lo == CY else y_lo
    hi = y_hi - BRK if y_hi == CY else y_hi
    ax.plot([BX, BX], [hi, lo], lw=0.75, color=INK, zorder=3)
    mid = (y_hi + y_lo) / 2
    ax.text(LBLX, mid + 0.95, name, ha="left", va="center",
            fontsize=6.5, color=INK, fontweight="bold")
    ax.text(LBLX, mid - 1.35, gloss, ha="left", va="center",
            fontsize=4.95, color="#4A5058")


axis_bracket(LANE_CY[0], LANE_CY[1], "$\\Delta_{\\mathrm{maj}}$",
             "beats the vote?")
axis_bracket(LANE_CY[1], LANE_CY[2], "SI", "does $\\mathcal{C}$ add value?")

# --- organising labels and the takeaway ------------------------------------
ax.text((1.2 + 54.0) / 2, FY1 + 1.55, "generate the candidate pool once",
        ha="center", va="center", fontsize=5.0, color=MUTED, style="italic")
ax.text((LX0 + LX1) / 2, FY1 + 1.55, "score three rules on that same pool",
        ha="center", va="center", fontsize=5.0, color=MUTED, style="italic")
ax.text((1.2 + 54.0) / 2, FY0 - 1.30,
        "the two axes are dissociable: neither implies the other",
        ha="center", va="center", fontsize=5.25, color=INK, style="italic")

out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                   "figures", "rf_pipeline.png"))
fig.savefig(out, dpi=400)          # no bbox_inches: keep the exact aspect
print(f"wrote {out}  aspect={ASPECT}  page height={DISP_W / ASPECT:.4f}in "
      f"({DISP_W / ASPECT / 0.1528:.2f} rendered lines)")
