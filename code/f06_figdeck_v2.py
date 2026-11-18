"""Figure deck, version 2: detailed framework figure and richer result figures.

Reuses the style and the simpler figures of f02_figdeck.py. New here:
  framework   five linked panels (acquisition, input views, models, evaluation, outputs), drawn with
              editable shapes, icons built from shapes, small native charts and data pictures
  roc         pooled and per-site ROC curves for each deep model
  strips      every development file score at its held-out site, with the fold threshold
  tsne        t-SNE of the frozen AST embedding against the raw spectrum
  nightmap    night score of every development night, by deployment, ordered by tier
  confusion   confusion tiles, development and locked (supplementary)
Every number comes from figures/figdata.json or figures/figdata_extra.json.
Writes figures/Paper5_Figures.pptx.
"""
import os
import json
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.chart.data import XyChartData, CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_MARKER_STYLE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE

import p5config as C
import f02_figdeck as B
from f02_figdeck import (rgb, text, box, arrow, caption, panel, base_chart, style_axis, marker, fill_series, fill_point,
                         gap, err_bars, x_at_bottom, INK, INK2, MUTED, GRID, AST, CNN, LSTM, LEAK, NOLEAK, TIER, FILL_SOFT,
                         MODEL_COL, FD)

FX = json.load(open(os.path.join(C.FIGURES, "figdata_extra.json")))
FIG = C.FIGURES
GREEN, ORANGE, BLUE, PURPLE, RED = "2E8B57", "D9822B", "2A78D6", "6250D6", "C0392B"


# ------------------------------------------------------------------ drawing helpers
def container(s, x, y, w, h, title, color, fill="FFFFFF"):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.04
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    sh.line.color.rgb = rgb(color); sh.line.width = Pt(1.75); sh.line.dash_style = MSO_LINE.DASH
    sh.shadow.inherit = False
    text(s, x + 0.12, y + 0.04, w - 0.2, 0.4, title, size=18, bold=True, color=color)
    return sh


def shape(s, kind, x, y, w, h, fill=None, line=None, lw=1.0, rot=0, dash=None):
    sh = s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    else:
        sh.fill.background()
    if line:
        sh.line.color.rgb = rgb(line); sh.line.width = Pt(lw)
        if dash:
            sh.line.dash_style = dash
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    sh.rotation = rot
    return sh


def line(s, x1, y1, x2, y2, color=INK2, w=1.0, dash=None):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color); c.line.width = Pt(w)
    if dash:
        c.line.dash_style = dash
    return c


def block_arrow(s, x, y, w, h, direction="right", color="9A9994"):
    kind = {"right": MSO_SHAPE.RIGHT_ARROW, "down": MSO_SHAPE.DOWN_ARROW, "left": MSO_SHAPE.LEFT_ARROW}[direction]
    return shape(s, kind, x, y, w, h, fill=color)


def padlock(s, x, y, size=0.32, color=RED):
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x + size * 0.18, y, size * 0.64, size * 0.75, line=color, lw=2.25)
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y + size * 0.42, size, size * 0.62, fill=color)


def droplet(s, x, y, size=0.16, color="3D7CC9"):
    shape(s, MSO_SHAPE.TEAR, x, y, size, size, fill=color, rot=-45)


def stopwatch(s, x, y, size=0.42, color=INK2):
    shape(s, MSO_SHAPE.RECTANGLE, x + size * 0.42, y - size * 0.12, size * 0.16, size * 0.12, fill=color)
    shape(s, MSO_SHAPE.OVAL, x, y, size, size, fill="FFFFFF", line=color, lw=2)
    line(s, x + size / 2, y + size / 2, x + size / 2, y + size * 0.18, color=color, w=2)
    line(s, x + size / 2, y + size / 2, x + size * 0.75, y + size * 0.55, color=color, w=2)


def nodes(s, x, y, n, gapy, r=0.11, fill="FFFFFF", color=INK2):
    out = []
    for i in range(n):
        yy = y + i * gapy
        shape(s, MSO_SHAPE.OVAL, x, yy, r, r, fill=fill, line=color, lw=1.0)
        out.append((x + r / 2, yy + r / 2))
    return out


def connect(s, a, b, color="B8B7B2", w=0.5):
    for p in a:
        for q in b:
            line(s, p[0], p[1], q[0], q[1], color=color, w=w)


def mini_wave(s, x, y, w, h, wave):
    cd = XyChartData()
    sr = cd.add_series("w")
    dt = wave["t"][1] - wave["t"][0]
    for t, a, b in zip(wave["t"], wave["env_max"], wave["env_min"]):
        sr.add_data_point(t, a); sr.add_data_point(t + dt / 2, b)
    ch = base_chart(s, XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, cd, x, y, w, h)
    ch.series[0].format.line.color.rgb = rgb(BLUE); ch.series[0].format.line.width = Pt(0.5); ch.series[0].smooth = False
    lim = max(max(abs(v) for v in wave["env_max"] + wave["env_min"]), 1e-3)
    ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = -lim, lim
    ch.category_axis.minimum_scale, ch.category_axis.maximum_scale = 0, 10
    for ax in (ch.value_axis, ch.category_axis):
        ax.visible = False
        ax.has_major_gridlines = False
    return ch


def picture(s, name, x, y, w, h, border=None):
    p = s.shapes.add_picture(os.path.join(FIG, name), Inches(x), Inches(y), Inches(w), Inches(h))
    if border:
        p.line.color.rgb = rgb(border); p.line.width = Pt(1)
    return p


# ------------------------------------------------------------------ Figure 1, detailed framework
def fig_framework(prs, n):
    """All text 18 pt; words cut to fit (writing skill)."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    W = FD["waveforms"]
    T = 18
    # (1) field data
    container(s, 0.12, 0.1, 3.3, 4.05, "1  Field data", GREEN, fill="F7FBF8")
    shape(s, MSO_SHAPE.MOON, 3.0, 0.16, 0.22, 0.32, fill="F2C14E", rot=-20)
    picture(s, "photo_logger.png", 0.28, 0.58, 1.3, 1.3, border=GREEN)
    box(s, 1.72, 0.58, 1.58, 0.72, "4,096 Hz\n10 s files", fill="FFFFFF", line=GREEN, size=T, bold=True)
    box(s, 1.72, 1.42, 1.58, 0.46, "12 sites", fill="FFFFFF", line=GREEN, size=T, bold=True)
    mini_wave(s, 0.28, 2.0, 3.0, 0.5, W[0])
    text(s, 0.28, 2.47, 3.0, 0.35, "Leak record, 10 s", size=T, color=INK2, align=PP_ALIGN.CENTER)
    box(s, 0.28, 2.95, 3.0, 0.5, "Sites 1 to 9: 2,159 files", fill=FILL_SOFT["GREY"], line=GREEN, size=T)
    box(s, 0.28, 3.55, 3.0, 0.5, "Sites 10 to 12: 283 files", fill=FILL_SOFT["RED"], line=RED, size=T)
    padlock(s, 3.08, 3.4, 0.24)
    block_arrow(s, 3.46, 1.55, 0.3, 0.42)
    # (2) input views
    container(s, 3.8, 0.1, 3.0, 4.05, "2  Input views", ORANGE, fill="FFFAF4")
    box(s, 3.93, 0.55, 2.75, 0.45, "A: 16 kHz, 0 to 8 kHz", fill="FFFFFF", line=ORANGE, size=T, bold=True)
    picture(s, "thumb_A_leak.png", 3.95, 1.07, 1.3, 0.8)
    picture(s, "thumb_A_noleak.png", 5.35, 1.07, 1.3, 0.8)
    text(s, 3.95, 1.86, 1.3, 0.35, "leak", size=T, color=RED, align=PP_ALIGN.CENTER, bold=True)
    text(s, 5.35, 1.86, 1.3, 0.35, "no leak", size=T, color=INK2, align=PP_ALIGN.CENTER, bold=True)
    text(s, 3.93, 2.2, 2.75, 0.35, "Top band holds no data", size=T, color=INK2, align=PP_ALIGN.CENTER)
    box(s, 3.93, 2.62, 2.75, 0.45, "B: 0 to 2,048 Hz", fill="FFFFFF", line=ORANGE, size=T, bold=True)
    picture(s, "thumb_B_leak.png", 3.95, 3.17, 1.3, 0.75)
    picture(s, "thumb_B_noleak.png", 5.35, 3.17, 1.3, 0.75)
    block_arrow(s, 6.84, 1.55, 0.3, 0.42)
    # (3) models
    container(s, 7.18, 0.1, 6.04, 4.05, "3  Deep models", BLUE, fill="F6F9FD")
    text(s, 7.3, 0.47, 3.6, 0.38, "AST, pretrained on AudioSet", size=T, bold=True, color=BLUE)
    padlock(s, 10.95, 0.5, 0.24, color=BLUE)
    for i in range(4):                                     # patch grid
        for j in range(4):
            shape(s, MSO_SHAPE.RECTANGLE, 7.35 + j * 0.16, 1.0 + i * 0.16, 0.14, 0.14, fill=("F2C1A6" if i < 2 else "C9D9F0"), line="FFFFFF", lw=0.5)
    text(s, 7.15, 1.68, 1.0, 0.38, "Patches", size=T, color=INK2, align=PP_ALIGN.CENTER)
    arrow(s, 8.02, 1.3, 8.2, 1.3)
    for k in range(3, -1, -1):                             # encoder stack with depth
        shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 8.25 + k * 0.06, 0.95 - k * 0.06, 1.75, 1.15, fill=("EAF2FB" if k else "FFFFFF"), line=BLUE, lw=1.0)
    for i, t in enumerate(["Norm", "Attention", "MLP"]):
        box(s, 8.33, 1.0 + i * 0.36, 1.6, 0.33, t, fill="FFFFFF", line="9DBFEA", size=T)
    text(s, 8.1, 2.1, 2.2, 0.38, "12 frozen layers", size=T, color=INK2, align=PP_ALIGN.CENTER)
    arrow(s, 10.12, 1.45, 10.35, 1.45)
    for i in range(6):                                     # pooled vector
        shape(s, MSO_SHAPE.RECTANGLE, 10.4, 0.98 + i * 0.16, 0.18, 0.14, fill="9DBFEA", line="FFFFFF", lw=0.5)
    text(s, 10.2, 1.95, 0.6, 0.38, "768", size=T, color=INK2, align=PP_ALIGN.CENTER)
    a = [(10.59, 1.05 + i * 0.16) for i in range(6)]
    b = nodes(s, 10.95, 1.33, 1, 0.3, r=0.2, fill=BLUE, color=BLUE)
    connect(s, a, b, color="9DBFEA", w=0.75)
    text(s, 10.6, 1.6, 0.9, 0.38, "Probe", size=T, color=INK2, align=PP_ALIGN.CENTER)
    arrow(s, 11.2, 1.43, 11.45, 1.43)
    box(s, 11.5, 1.13, 1.6, 0.6, "Score p", fill=FILL_SOFT["AST"], line=BLUE, size=T, bold=True)
    text(s, 11.45, 1.78, 1.7, 0.38, "769 weights", size=T, color=INK2, align=PP_ALIGN.CENTER)
    line(s, 7.3, 2.52, 13.1, 2.52, color="D9D8D3", w=1, dash=MSO_LINE.DASH)
    # CNN row
    text(s, 7.3, 2.56, 2.6, 0.38, "CNN, from scratch", size=T, bold=True, color=CNN)
    for i, sz in enumerate([0.58, 0.5, 0.42, 0.34]):
        x = 9.7 + i * 0.38
        shape(s, MSO_SHAPE.CUBE, x, 2.58 + (0.58 - sz) / 2, sz * 0.7, sz, fill="FBD8C5", line=CNN, lw=1)
    arrow(s, 11.25, 2.88, 11.45, 2.88)
    box(s, 11.5, 2.6, 1.6, 0.55, "Score p", fill=FILL_SOFT["CNN"], line=CNN, size=T, bold=True)
    text(s, 7.3, 2.95, 3.0, 0.38, "4 blocks, 0.42 M weights", size=T, color=INK2)
    line(s, 7.3, 3.42, 13.1, 3.42, color="D9D8D3", w=1, dash=MSO_LINE.DASH)
    # LSTM row
    text(s, 7.3, 3.5, 2.4, 0.38, "LSTM, from scratch", size=T, bold=True, color="13865C")
    cells = []
    for i in range(4):
        x = 9.95 + i * 0.32
        shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 3.55, 0.24, 0.3, fill="D4F0E5", line=LSTM, lw=1)
        cells.append(x)
    for x0, x1 in zip(cells[:-1], cells[1:]):
        line(s, x0 + 0.24, 3.65, x1, 3.65, color=LSTM, w=1.25)
        line(s, x0 + 0.24, 3.75, x1, 3.75, color=LSTM, w=1.25)
    arrow(s, 11.25, 3.7, 11.45, 3.7)
    box(s, 11.5, 3.47, 1.6, 0.5, "Score p", fill=FILL_SOFT["LSTM"], line=LSTM, size=T, bold=True)
    # down arrows
    block_arrow(s, 1.55, 4.17, 0.42, 0.3, "down")
    block_arrow(s, 9.95, 4.17, 0.42, 0.3, "down")
    # (4) evaluation
    container(s, 0.12, 4.5, 6.45, 2.33, "4  Evaluation without site leakage", PURPLE, fill="F8F7FD")
    cs = 0.17
    for f in range(9):
        for k in range(9):
            shape(s, MSO_SHAPE.RECTANGLE, 0.85 + k * cs, 5.0 + f * cs, cs - 0.02, cs - 0.02,
                  fill=(ORANGE if f == k else "D6D2F2"), line=None)
    text(s, 0.15, 5.35, 0.7, 0.7, "9\nfolds", size=T, color=INK2, align=PP_ALIGN.CENTER)
    shape(s, MSO_SHAPE.RECTANGLE, 2.5, 5.03, 0.15, 0.15, fill=ORANGE)
    text(s, 2.68, 4.92, 1.75, 0.36, "Held-out site", size=T, color=INK)
    shape(s, MSO_SHAPE.RECTANGLE, 2.5, 5.38, 0.15, 0.15, fill="D6D2F2")
    text(s, 2.68, 5.27, 1.75, 0.36, "Training sites", size=T, color=INK)
    box(s, 2.5, 5.72, 1.95, 1.0, "Inner loop sets\nhead and\nthreshold", fill="FFFFFF", line=PURPLE, size=T)
    for i in range(3):                                     # bootstrap icon
        for j in range(4):
            shape(s, MSO_SHAPE.OVAL, 4.65 + j * 0.2, 4.98 + i * 0.18, 0.14, 0.14, fill=(LEAK if (i + j) % 3 == 0 else NOLEAK))
    shape(s, MSO_SHAPE.CIRCULAR_ARROW, 5.55, 4.9, 0.5, 0.5, fill=PURPLE)
    text(s, 4.6, 5.52, 1.95, 0.66, "2,000 deployment\nresamples", size=T, color=INK)
    padlock(s, 4.62, 6.3, 0.24)
    text(s, 4.92, 6.18, 1.62, 0.62, "Sites 10 to 12:\nscored once", size=T, color=RED, bold=True)
    block_arrow(s, 6.6, 5.45, 0.3, 0.42)
    # (5) outputs
    container(s, 6.92, 4.5, 6.3, 2.33, "5  Outputs", RED, fill="FDF7F7")
    text(s, 7.02, 4.9, 1.6, 0.36, "File score", size=T, bold=True)
    for i in range(10):
        c = ["9A9994", "A49C98", "AE9B97", "B99A95", "C49893", "CF8E89", "D97F7A", "E16C69", "E35A58", "E34948"][i]
        shape(s, MSO_SHAPE.RECTANGLE, 7.08 + i * 0.14, 5.4, 0.14, 0.24, fill=c)
    thr = FD["locked_cuts"]["threshold"]
    shape(s, MSO_SHAPE.ISOSCELES_TRIANGLE, 7.08 + thr * 1.4 - 0.07, 5.68, 0.14, 0.12, fill=INK)
    text(s, 7.02, 5.85, 1.6, 0.66, f"Threshold\n{thr:.2f}", size=T, color=INK2)
    ig = FD["ig_contrast"]
    text(s, 8.6, 4.9, 1.8, 0.36, "Band evidence", size=T, bold=True)
    vals = [("<2k", ig["fraction_of_contrast_in_band"]["est"], BLUE), ("edge", ig["fraction_of_contrast_straddle"]["est"], "9DBFEA"),
            (">2k", ig["fraction_of_contrast_dead"]["est"], "C9C8C3")]
    base_y = 6.12
    for i, (lab, v, c) in enumerate(vals):
        hgt = abs(v) * 0.5
        x = 8.7 + i * 0.58
        shape(s, MSO_SHAPE.RECTANGLE, x, base_y - hgt if v > 0 else base_y, 0.4, max(hgt, 0.02), fill=c)
        text(s, x - 0.15, 6.44, 0.7, 0.36, lab, size=T, color=INK2, align=PP_ALIGN.CENTER)
        text(s, x - 0.15, (base_y - hgt - 0.36) if v > 0 else base_y + hgt, 0.7, 0.36, f"{v:+.0%}", size=T, color=INK, align=PP_ALIGN.CENTER)
    line(s, 8.6, base_y, 10.35, base_y, color=INK2, w=0.75)
    text(s, 10.48, 4.9, 1.8, 0.36, "Evidence tier", size=T, bold=True)
    names = ["Major", "Significant", "Moderate", "Borderline", "Not flagged"]
    for i, nm in enumerate(names):
        y = 5.3 + i * 0.29
        shape(s, MSO_SHAPE.RECTANGLE, 10.5, y, 1.3, 0.27, fill=TIER[4 - i])
        text(s, 10.45, y - 0.04, 1.4, 0.35, nm, size=T, color=("FFFFFF" if i < 2 else INK), align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        for d in range(4 - i):
            droplet(s, 11.86 + d * 0.13, y + 0.06, 0.12)
    text(s, 12.42, 4.9, 0.8, 0.36, "Cost", size=T, bold=True)
    stopwatch(s, 12.6, 5.38, 0.42)
    c = FD["cost"]["ast_A_pool_LP"]
    text(s, 12.36, 5.9, 0.88, 0.66, f"{c['cpu1_ms'] / 1000:.1f} s\nper file", size=T, color=INK2, align=PP_ALIGN.CENTER)
    caption(s, n, "Research framework")


# ------------------------------------------------------------------ result figures
def fig_roc(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    for j, nm in enumerate(("AST", "CNN", "LSTM")):
        R = FX["roc"][nm]
        x = 0.35 + j * 4.3
        panel(s, x - 0.05, 0.2, f"({'abc'[j]})")
        cd = XyChartData()
        for site, d in sorted(R["sites"].items(), key=lambda kv: int(kv[0])):
            sr = cd.add_series(f"Site {site}")
            for a, b in zip(d["fpr"], d["tpr"]):
                sr.add_data_point(a, b)
        sr = cd.add_series("Pooled")
        for a, b in zip(R["fpr"], R["tpr"]):
            sr.add_data_point(a, b)
        dg = cd.add_series("Chance")
        dg.add_data_point(0, 0); dg.add_data_point(1, 1)
        ch = base_chart(s, XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, cd, x, 0.5, 4.2, 5.4)
        k = len(R["sites"])
        for i, sr in enumerate(ch.series):
            sr.smooth = False
            if i < k:
                sr.format.line.color.rgb = rgb("C9C8C3"); sr.format.line.width = Pt(1.0)
            elif i == k:
                sr.format.line.color.rgb = rgb(MODEL_COL[nm]); sr.format.line.width = Pt(3.0)
            else:
                sr.format.line.color.rgb = rgb(INK2); sr.format.line.width = Pt(1.0); sr.format.line.dash_style = MSO_LINE.DASH
        style_axis(ch.value_axis, "True positive rate" if j == 0 else None, size=14, minv=0, maxv=1, major=0.2, fmt="0.0", grid=True)
        style_axis(ch.category_axis, "False positive rate", size=14, minv=0, maxv=1, major=0.2, fmt="0.0")
        text(s, x + 1.55, 3.75, 2.6, 0.7, f"{nm}\npooled AUC {R['auc']:.3f}", size=18, bold=True, color=MODEL_COL[nm] if nm != "LSTM" else "13865C")
    text(s, 0.4, 6.05, 12.5, 0.5, "Thick line: all held-out files pooled. Thin grey lines: one per held-out site with both classes (8 sites).",
         size=14, color=INK2)
    caption(s, n, "ROC curves under leave-one-site-out testing")


def fig_strips(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    for j, nm in enumerate(("AST", "CNN", "LSTM")):
        S_ = FX["strips"][nm]
        y0 = 0.15 + j * 2.2
        panel(s, 0.2, y0, f"({'abc'[j]})")
        cd = XyChartData()
        nl = cd.add_series("No-leak file"); lk = cd.add_series("Leak file"); th = cd.add_series("Threshold")
        for site, yv, p, jt in zip(S_["site"], S_["y"], S_["p"], S_["jitter"]):
            (lk if yv == 1 else nl).add_data_point(site + jt, p)
        thr = {}
        for site, t in zip(S_["site"], S_["thr"]):
            thr[site] = t
        for site in sorted(thr):
            th.add_data_point(site, thr[site])
        ch = base_chart(s, XL_CHART_TYPE.XY_SCATTER, cd, 0.5, y0 + 0.05, 12.5, 2.2, legend=(j == 0))
        if j == 0:
            ch.legend.position = XL_LEGEND_POSITION.TOP
        marker(ch.series[0], NOLEAK, size=3, style=XL_MARKER_STYLE.CIRCLE)
        marker(ch.series[1], LEAK, size=3, style=XL_MARKER_STYLE.CIRCLE)
        for sr in (ch.series[0], ch.series[1]):
            sr.marker.format.line.fill.background()
        marker(ch.series[2], INK, size=20, style=XL_MARKER_STYLE.DASH)
        style_axis(ch.value_axis, f"{nm} score", size=13, minv=0, maxv=1, major=0.5, fmt="0.0", grid=True)
        style_axis(ch.category_axis, "Held-out development site" if j == 2 else None, size=13, minv=0.4, maxv=9.6, major=1,
                   fmt='[<0.5]"";[>9.5]"";0')
    caption(s, n, "Every development file score at its held-out site")


def fig_tsne(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    T = FX["tsne"]
    for j, (key, title) in enumerate((("spectrum", "Raw spectrum, 0 to 2,048 Hz, no model"), ("ast", "Frozen AST embedding, input A"))):
        x = 0.35 + j * 6.45
        panel(s, x - 0.05, 0.2, f"({'ab'[j]})")
        text(s, x + 0.45, 0.2, 5.8, 0.4, title, size=16, bold=True)
        cd = XyChartData()
        nl = cd.add_series("No-leak file"); lk = cd.add_series("Leak file")
        for (a, b), yv in zip(T[key], T["y"]):
            (lk if yv == 1 else nl).add_data_point(a, b)
        ch = base_chart(s, XL_CHART_TYPE.XY_SCATTER, cd, x, 0.6, 6.2, 5.9, legend=(j == 1))
        if j == 1:
            ch.legend.position = XL_LEGEND_POSITION.BOTTOM
        marker(ch.series[0], NOLEAK, size=4); marker(ch.series[1], LEAK, size=4)
        for sr in ch.series:
            sr.marker.format.line.fill.background()
        for ax in (ch.value_axis, ch.category_axis):
            ax.visible = False
            ax.has_major_gridlines = False
    text(s, 0.4, 6.45, 12.5, 0.4, "t-SNE of 2,159 development files (perplexity 30). Only neighbourhoods are meaningful; axes have no units.",
         size=13, color=INK2)
    caption(s, n, "Two-dimensional maps of the development files")


def heat(v):
    """Score 0..1 to a grey-to-red ramp (no-leak grey at 0, leak red at 1)."""
    a = np.array([0xEE, 0xED, 0xEA]); b = np.array([0xB2, 0x1F, 0x24])
    c = (a + (b - a) * float(np.clip(v, 0, 1))).astype(int)
    return "%02X%02X%02X" % tuple(c)


def fig_nightmap(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    N = FX["nights"]
    maxn = max(len(r["nights"]) for r in N)
    x0, y0 = 2.55, 0.6
    cw = min(0.2, 8.75 / maxn)
    ch = 5.95 / len(N)
    text(s, 0.1, 0.12, 1.5, 0.4, "Tier", size=18, bold=True)
    text(s, 1.62, 0.12, 0.9, 0.4, "Label", size=18, bold=True)
    text(s, x0, 0.12, 8.7, 0.4, "Nights in date order: mean AST score of the night", size=18, bold=True)
    tier_col = {"Not flagged": TIER[0], "Borderline": TIER[1], "Moderate": TIER[2], "Significant": TIER[3], "Major": TIER[4]}
    prev = None
    for i, r in enumerate(N):
        y = y0 + i * ch
        if r["tier"] != prev:
            text(s, 0.1, y - 0.04, 1.5, 0.38, r["tier"], size=18, color=INK)
            if prev is not None:
                line(s, 0.1, y - 0.005, x0 + maxn * cw, y - 0.005, color=INK2, w=0.75)
            prev = r["tier"]
        shape(s, MSO_SHAPE.RECTANGLE, 1.62, y, 0.2, ch * 0.9, fill=tier_col[r["tier"]])
        shape(s, MSO_SHAPE.RECTANGLE, 1.95, y, 0.35, ch * 0.9, fill=(LEAK if r["y"] == 1 else NOLEAK))
        for k, v in enumerate(r["nights"]):
            shape(s, MSO_SHAPE.RECTANGLE, x0 + k * cw, y, cw * 0.92, ch * 0.9, fill=heat(v),
                  line=(INK if v >= r["thr"] else None), lw=0.5)
    kx = x0 + maxn * cw + 0.25
    text(s, kx, 0.6, 1.9, 0.38, "Night score", size=18, bold=True)
    for i in range(11):
        v = 1 - i / 10
        shape(s, MSO_SHAPE.RECTANGLE, kx, 1.05 + i * 0.2, 0.3, 0.2, fill=heat(v))
        if i % 5 == 0:
            text(s, kx + 0.35, 0.98 + i * 0.2, 0.7, 0.36, f"{v:.1f}", size=18, color=INK2)
    shape(s, MSO_SHAPE.RECTANGLE, kx, 3.55, 0.3, 0.22, fill=heat(0.6), line=INK, lw=0.75)
    text(s, kx + 0.35, 3.45, 1.5, 0.7, "Outlined:\nabove τ", size=18, color=INK2)
    shape(s, MSO_SHAPE.RECTANGLE, kx, 4.4, 0.3, 0.22, fill=LEAK)
    text(s, kx + 0.35, 4.32, 1.5, 0.38, "Leak", size=18, color=INK2)
    shape(s, MSO_SHAPE.RECTANGLE, kx, 4.82, 0.3, 0.22, fill=NOLEAK)
    text(s, kx + 0.35, 4.74, 1.5, 0.38, "No leak", size=18, color=INK2)
    caption(s, n, "Night-by-night scores of every development deployment")


def figS_confusion(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    Cf = FX["confusion"]
    for i, (part, title) in enumerate((("dev", "Leave one site out, Sites 1 to 9"), ("locked", "Locked test, Sites 10 to 12"))):
        y = 0.3 + i * 3.25
        text(s, 0.3, y, 6, 0.4, title, size=16, bold=True)
        for j, nm in enumerate(("AST", "CNN", "LSTM")):
            c = Cf[nm][part]
            x = 0.6 + j * 4.2
            text(s, x, y + 0.45, 3.6, 0.35, nm, size=15, bold=True, color=MODEL_COL[nm] if nm != "LSTM" else "13865C", align=PP_ALIGN.CENTER)
            tot = sum(c.values())
            cells = [("TN", c["TN"], 0, 0), ("FP", c["FP"], 0, 1), ("FN", c["FN"], 1, 0), ("TP", c["TP"], 1, 1)]
            mx = max(v for _, v, _, _ in cells)
            for lab, v, r_, k in cells:
                good = lab in ("TN", "TP")
                fillc = MODEL_COL[nm] if good else "EEEDEA"
                box(s, x + 0.6 + k * 1.5, y + 0.85 + r_ * 1.0, 1.45, 0.95, f"{lab}\n{v:,}", fill=fillc, line="FFFFFF", size=14,
                    bold=True, color=("FFFFFF" if good else INK))
            text(s, x - 0.4, y + 1.12, 0.98, 0.38, "No leak", size=18, color=INK2, align=PP_ALIGN.RIGHT)
            text(s, x - 0.4, y + 2.12, 0.98, 0.38, "Leak", size=18, color=INK2, align=PP_ALIGN.RIGHT)
            text(s, x + 0.6, y + 2.8, 1.45, 0.38, "No leak", size=18, color=INK2, align=PP_ALIGN.CENTER)
            text(s, x + 2.1, y + 2.8, 1.45, 0.38, "Leak", size=18, color=INK2, align=PP_ALIGN.CENTER)
        if i == 0:
            text(s, 6.6, y, 6.6, 0.4, "Rows: true class. Columns: predicted class.", size=18, color=INK2, align=PP_ALIGN.RIGHT)
    caption(s, n, "Decisions at the fixed thresholds", supp=True)


MIN_PT = 18   # writing skill: no figure text below 18 pt


def floor_fonts(prs):
    """Raises every run, paragraph default and chart text below MIN_PT to MIN_PT (slides and embedded charts)."""
    A = "http://schemas.openxmlformats.org/drawingml/2006/main"
    tags = {"{%s}%s" % (A, t) for t in ("rPr", "defRPr", "endParaRPr")}

    def fix(root):
        for el in root.iter():
            if el.tag in tags and el.get("sz") is not None and int(el.get("sz")) < MIN_PT * 100:
                el.set("sz", str(MIN_PT * 100))
    for s in prs.slides:
        fix(s._element)
        for sh in s.shapes:
            if getattr(sh, "has_chart", False) and sh.has_chart:
                fix(sh.chart._chartSpace)


def main():
    import f08_method_figs as M
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    main_figs = [fig_framework, B.fig_dataset, M.fig_m_ast, M.fig_m_scratch, M.fig_m_eval, B.fig_signals, B.fig_main, fig_roc, fig_strips, B.fig_sites,
                 B.fig_frequency, B.fig_inputs, B.fig_layers, fig_tsne, fig_nightmap, B.fig_tiers, B.fig_locked]
    for i, f in enumerate(main_figs, 1):
        f(prs, i)
    supp = [B.figS_heads, B.figS_protocols, B.figS_effects, B.figS_igbins, B.figS_tones, figS_confusion]
    for i, f in enumerate(supp, 1):
        f(prs, i)
    floor_fonts(prs)
    B.set_meta(prs, "this study figures")
    prs.save(B.OUT)
    print("saved", B.OUT, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
