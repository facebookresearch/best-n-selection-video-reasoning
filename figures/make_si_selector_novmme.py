#!/usr/bin/env python3
"""Appendix figure: Selection Increment (SI) per benchmark, weak vs. strong verifier.

Three benchmarks carry both deployed verifiers. The stronger GPT-5.6 Sol has positive
SI on all three; the weaker Muse holistic-48 splits in sign. Numbers verified in
analysis_handoff/gpt_headroom_test_2026-08-08.md, verified_mlvu_test_2026-08-07.md and
verified_vrbench_2026-08-08.md. Writes rf_si_selector_novmme.png; leaves the shared
rf_si_selector.png used by paper.tex untouched.

Drawing rules, same three as the taxonomy figure:
  1. Canvas width == display width (0.82 * 5.5in), so fontsize == rendered pt. The
     previous version was drawn 6.4in wide and shown at 4.51in -- a 1.42x downscale
     that put its 11pt axis label on the page at 7.7pt and its 7.5pt value labels at
     5.3pt.
  2. No dead range. The old version reserved y up to 12.6 for a 11.16 maximum and a
     -2.4 floor for a -1.08 minimum.
  3. dpi 400, and NO bbox_inches="tight" -- tight cropping silently changes the aspect,
     which is what makes rendered point sizes unpredictable in the first place.

Orientation changed to horizontal. At this aspect (2.35) the vertical form spends its
short height on empty headroom above the bars; laid on its side, the signed quantity
reads left/right off the zero rule, the benchmark names sit unrotated in the tick
gutter, and the legend occupies the space the shortest pair leaves free.

Deliberately NOT plotted: standalone accuracy. The caption's inverse-tracking claim is
made in the text against verified per-arm accuracies; the values implied here by
selection minus SI are derived from rounded inputs and are not a verified digest, so
they are not drawn.
"""
import os

for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(v, None)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "text.usetex": False,
    "axes.linewidth": 0.7,
})

# --- geometry ---------------------------------------------------------------
DISP_W = 0.82 * 5.5
ASPECT = 2.35
FIG_W, FIG_H = DISP_W, DISP_W / ASPECT

BLUE = "#0064E0"   # Muse holistic-48 (weaker)
RED = "#D6372B"    # GPT-5.6 Sol (stronger)
INK = "#15181C"
GRID = "#9AA0A8"
FAINT = "#DCDFE3"

# Ordered by Muse SI, ascending. Verified SI (pt).
benches = ["TempCompass", "VRBench", "MLVU-Test"]
si_muse = [-1.08, +3.67, +5.68]
si_gpt = [+1.15, +3.96, +11.16]

TS, TK, VL = 6.4, 6.0, 5.9   # axis label / ticks / value labels
y = np.arange(len(benches))
h = 0.30
off = 0.17

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
XLO, XHI = -2.0, 12.7
ax.set_xlim(XLO, XHI)
ax.set_ylim(len(benches) - 0.55, -0.55)     # inverted: first bench on top

# faint vertical guides behind the bars; the zero rule is the emphasized one
for gx in (4, 8, 12):
    ax.axvline(gx, color=FAINT, lw=0.6, zorder=0)
ax.axvline(0, color=GRID, lw=0.8, zorder=1)

ax.barh(y - off, si_muse, h, color=BLUE, zorder=3, linewidth=0)
ax.barh(y + off, si_gpt, h, color=RED, zorder=3, linewidth=0)

# value labels, hung off the far end of each bar
for yi, v in zip(y - off, si_muse):
    pad = 0.22 if v >= 0 else -0.22
    ax.text(v + pad, yi, f"{v:+.1f}".replace("-", "\u2212"), fontsize=VL,
            color=BLUE, va="center",
            ha="left" if v >= 0 else "right", zorder=4)
for yi, v in zip(y + off, si_gpt):
    ax.text(v + 0.22, yi, f"{v:+.1f}".replace("-", "\u2212"), fontsize=VL,
            color=RED, va="center", ha="left", zorder=4)

# --- inline legend, in the gap the shortest pair leaves at top right --------
# order matches the within-group bar order: Muse is the upper bar
ax.text(XHI - 0.15, -0.46, "Muse holistic-48", fontsize=VL, color=BLUE,
        ha="right", va="top")
ax.text(XHI - 0.15, 0.10, "GPT-5.6 Sol", fontsize=VL, color=RED,
        ha="right", va="top")

ax.set_yticks(y)
ax.set_yticklabels(benches, fontsize=TK, color=INK)
ax.set_xticks([0, 4, 8, 12])
ax.set_xlabel("Selection Increment  (selection $-$ standalone, pt)",
              fontsize=TS, color=INK, labelpad=1.5)
ax.tick_params(axis="x", labelsize=TK, colors=INK, length=2.2, width=0.6, pad=1.5)
ax.tick_params(axis="y", length=0, pad=2.0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)

fig.subplots_adjust(left=0.155, right=0.995, top=0.97, bottom=0.215)
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                   "figures", "rf_si_selector_novmme.png"))
fig.savefig(out, dpi=400)          # no bbox_inches: keep the exact aspect
print(f"wrote {out}  aspect={ASPECT}  page height={DISP_W / ASPECT:.4f}in")
