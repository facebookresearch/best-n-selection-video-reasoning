#!/usr/bin/env python3
"""Taxonomy figure: Selection Increment (x) against gain over self-consistency (y).

The figure's one job is to show that the two axes are DISSOCIABLE -- a cell can have
positive SI and still lose to the free majority vote. Three drawing rules make that
readable at publication size; the previous version broke all three.

  1. The canvas is drawn at exactly the width it is displayed at (0.72 * 5.5in of the
     NeurIPS text block), so a fontsize of 6.2 here is 6.2pt on the page. The previous
     version was drawn 7.2in wide and shown at 3.96in -- a 1.82x downscale that
     rendered its 11pt axis labels at 6.0pt and its 8pt point labels at 4.4pt.
  2. Ranges are padded to the data, not to round numbers 3 points past it. The old
     version spent roughly a third of its canvas on empty margin.
  3. Each benchmark is named ONCE, on its connector, instead of once per endpoint;
     the marker shape carries the verifier, as the legend says. This halves the text
     without losing anything.

The quadrant tags state the two conditions and nothing else -- the old parenthetical
glosses ("selection real & useful", "helps the verifier, not useful") repeated the
caption verbatim. The legend sits in the lower-left quadrant, which holds no data.

ASPECT is the page-budget knob: height on the page = 0.72 * 5.5 / ASPECT inches. The
figure this replaces was 2.7233in tall (17.82 rendered lines at 0.1528in/line); at
ASPECT 2.05 this one is 1.9317in (12.64 lines), returning ~5.2 lines to the main text.
That matters because the main text is currently 3 lines over the 8-page workshop limit.
"""
import os

for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(v, None)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- typography: STIXGeneral is Times-metric-compatible, matching NeurIPS ----
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "text.usetex": False,
    "axes.linewidth": 0.7,
})

# --- geometry ---------------------------------------------------------------
DISP_W = 0.72 * 5.5          # inches the float occupies on the page
ASPECT = 2.05                # width / height
FIG_W, FIG_H = DISP_W, DISP_W / ASPECT

# --- palette ---------------------------------------------------------------
BLUE = "#0064E0"     # Muse holistic-48
RED = "#D6372B"      # GPT-5.6 Sol
INK = "#15181C"
GRID = "#9AA0A8"
LINK = "#C4C8CD"
GREEN_T = "#1A7F45"  # quadrant tag: SI>0 and beats the vote
AMBER_T = "#B26A00"  # quadrant tag: SI<0 and beats the vote
RED_T = "#C0392B"    # quadrant tag: SI>0 and loses to the vote

# bench -> (label, muse_SI, muse_D, gpt_SI, gpt_D, (label_dx, label_dy), ha)
DATA = {
    "TempCompass": ("TempCompass", -1.08, 7.94, 1.15, 9.43, (-0.30, 1.42), "center"),
    "MLVU-Test":   ("MLVU-Test",    5.68, 0.81, 11.16, -1.99, (-0.25, 1.95), "center"),
    "VRBench":     ("VRBench",      3.67, -3.26, 3.96, -1.62, (-0.55, 0.10), "right"),
}

TS, LS, QS = 6.2, 5.8, 5.4   # axis-label / point-label / quadrant-tag sizes

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
XLO, XHI, YLO, YHI = -2.5, 12.7, -4.4, 11.1
ax.set_xlim(XLO, XHI)
ax.set_ylim(YLO, YHI)

# --- quadrant tints, keyed to the two zero lines ---------------------------
x0 = (0 - XLO) / (XHI - XLO)
ax.axhspan(0, YHI, xmin=x0, xmax=1, color=GREEN_T, alpha=0.055, zorder=0)
ax.axhspan(0, YHI, xmin=0, xmax=x0, color=AMBER_T, alpha=0.065, zorder=0)
ax.axhspan(YLO, 0, xmin=x0, xmax=1, color=RED_T, alpha=0.05, zorder=0)
ax.axhline(0, color=GRID, lw=0.8, zorder=1)
ax.axvline(0, color=GRID, lw=0.8, zorder=1)

# --- the six cells, three linked pairs -------------------------------------
for _, (lab, ms, md, gs, gd, (ldx, ldy), ha) in DATA.items():
    ax.plot([ms, gs], [md, gd], color=LINK, lw=1.0, zorder=2,
            solid_capstyle="round")
    ax.scatter([ms], [md], s=26, color=BLUE, zorder=4, linewidths=0)
    ax.scatter([gs], [gd], s=26, color=RED, marker="s", zorder=4, linewidths=0)
    # name the pair once, at the connector midpoint
    ax.text((ms + gs) / 2 + ldx, (md + gd) / 2 + ldy, lab, fontsize=LS,
            color=INK, ha=ha, va="center", zorder=5)

# --- quadrant tags: the two conditions, no gloss ---------------------------
ax.text(XHI - 0.25, YHI - 0.30, "SI $>$ 0, beats the vote", fontsize=QS,
        color=GREEN_T, ha="right", va="top")
# the SI<0 band is narrow, so its tag goes at the FOOT of the band, where the
# band is empty -- at the top it collides with the TempCompass pair label
ax.text(XLO + 0.25, 0.35, "SI $<$ 0,\nbeats the vote", fontsize=QS,
        color=AMBER_T, ha="left", va="bottom", linespacing=1.35)
ax.text(XHI - 0.25, YLO + 0.28, "SI $>$ 0, loses to the vote", fontsize=QS,
        color=RED_T, ha="right", va="bottom")

# --- legend, placed in the empty lower-left quadrant -----------------------
ax.scatter([], [], s=26, color=BLUE, linewidths=0, label="Muse holistic-48")
ax.scatter([], [], s=26, color=RED, marker="s", linewidths=0, label="GPT-5.6 Sol")
leg = ax.legend(loc="lower left", frameon=False, fontsize=LS,
                handletextpad=0.45, borderpad=0.15, labelspacing=0.35,
                borderaxespad=0.5)
for t in leg.get_texts():
    t.set_color(INK)

# --- axes -------------------------------------------------------------------
ax.set_xlabel("Selection Increment  SI  (pt)", fontsize=TS, color=INK,
              labelpad=1.5)
ax.set_ylabel("$\\Delta$ over majority@8  (pt)", fontsize=TS, color=INK,
              labelpad=1.5)
ax.set_xticks([0, 4, 8, 12])
ax.set_yticks([-4, 0, 4, 8])
ax.tick_params(axis="both", labelsize=QS, colors=INK, length=2.2, width=0.6,
               pad=1.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)

fig.subplots_adjust(left=0.115, right=0.995, top=0.985, bottom=0.145)
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                   "figures", "rf_si_taxonomy_novmme.png"))
fig.savefig(out, dpi=400)          # no bbox_inches: keep the exact aspect
print(f"wrote {out}  aspect={ASPECT}  page height={DISP_W / ASPECT:.4f}in "
      f"({DISP_W / ASPECT / 0.1528:.2f} rendered lines)")
