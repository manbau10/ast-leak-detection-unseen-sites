"""Figure deck for this study. One editable PowerPoint: native charts, shapes and text boxes.

Reads figures/figdata.json (f01_figdata.py) only. Writes figures/Paper5_Figures.pptx.
Style: Times New Roman; chart text 16 pt; one validated categorical palette
(AST blue, CNN orange, LSTM aqua; dataviz check passed, aqua carries direct labels);
light horizontal grid only; 95% CIs as custom error bars; caption "Figure N: Title" below each figure.
Deep-learning models only (decision of 6 Nov 2026).
"""
import os
import json
import copy
import numpy as np
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.enum.chart import (XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION, XL_MARKER_STYLE, XL_TICK_MARK,
                             XL_AXIS_CROSSES, XL_LABEL_POSITION)
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

import p5config as C

FD = json.load(open(os.path.join(C.FIGURES, "figdata.json")))
OUT = os.path.join(C.FIGURES, "Paper5_Figures.pptx")
FONT = "Times New Roman"
INK, INK2, MUTED, GRID = "0B0B0B", "52514E", "8A8984", "E4E3DF"
AST, CNN, LSTM = "2A78D6", "EB6834", "1BAF7A"
LEAK, NOLEAK = "E34948", "9A9994"
TIER = ["E9EEF6", "C3D5EE", "8DB2E3", "4A86D0", "1D4E8F"]          # sequential blue, light to dark
FILL_SOFT = {"AST": "EAF2FB", "CNN": "FDEEE7", "LSTM": "E6F6F0", "GREY": "F2F2F0", "GREEN": "E9F5EC", "RED": "FBEAEA"}
TXT = 16
MODEL_COL = {"AST": AST, "CNN": CNN, "LSTM": LSTM}


# ------------------------------------------------------------------ text and shapes
def rgb(h):
    return RGBColor.from_string(h)


def text(slide, x, y, w, h, s, size=TXT, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.03)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = anchor
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        f = r.font
        f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
        f.color.rgb = rgb(color)
    return tb


def box(slide, x, y, w, h, s="", fill="FFFFFF", line=INK2, size=TXT, bold=False, color=INK, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        lw=1.25, align=PP_ALIGN.CENTER):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = 0.12
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(fill)
    if line:
        sh.line.color.rgb = rgb(line)
        sh.line.width = Pt(lw)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line_ in enumerate(s.split("\n") if s else []):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line_
        r.font.name, r.font.size, r.font.color.rgb = FONT, Pt(size), rgb(color)
        r.font.bold = bold if i == 0 else False
    return sh


def arrow(slide, x1, y1, x2, y2, color=INK2, w=1.75):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(w)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle"); tail.set("w", "med"); tail.set("len", "med")
    return c


def caption(slide, n, title, supp=False):
    lab = f"Figure S{n}" if supp else f"Figure {n}"
    text(slide, 0.5, 6.85, 12.3, 0.5, f"{lab}: {title}", size=18, bold=True)


def panel(slide, x, y, s):
    text(slide, x, y, 0.6, 0.4, s, size=18, bold=True)


# ------------------------------------------------------------------ chart styling
def _font(tx_props_parent, size=TXT, color=INK):
    pass


def style_axis(ax, title=None, size=TXT, grid=False, fmt=None, minv=None, maxv=None, major=None, line=True):
    ax.has_major_gridlines = grid
    if grid:
        gl = ax.major_gridlines.format.line
        gl.color.rgb = rgb(GRID); gl.width = Pt(0.75)
    ax.tick_labels.font.name = FONT
    ax.tick_labels.font.size = Pt(size)
    ax.tick_labels.font.color.rgb = rgb(INK)
    if fmt:
        ax.tick_labels.number_format = fmt
        ax.tick_labels.number_format_is_linked = False
    if minv is not None:
        ax.minimum_scale = minv
    if maxv is not None:
        ax.maximum_scale = maxv
    if major is not None:
        ax.major_unit = major
    ax.format.line.color.rgb = rgb(INK2 if line else "FFFFFF")
    ax.format.line.width = Pt(1)
    if not line:
        ax.format.line.fill.background()
    ax.major_tick_mark = XL_TICK_MARK.OUTSIDE if line else XL_TICK_MARK.NONE
    ax.minor_tick_mark = XL_TICK_MARK.NONE
    if title:
        ax.has_title = True
        tf = ax.axis_title.text_frame
        tf.text = title
        for p in tf.paragraphs:
            for r in p.runs:
                r.font.name, r.font.size, r.font.bold = FONT, Pt(size), False
                r.font.color.rgb = rgb(INK)


def base_chart(slide, kind, data, x, y, w, h, legend=False):
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), data)
    ch = gf.chart
    ch.has_title = False
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.name = FONT
        ch.legend.font.size = Pt(TXT)
    ch.font.name = FONT
    ch.font.size = Pt(TXT)
    # no chart border, transparent background
    cs = ch._chartSpace
    sppr = cs.find(qn("c:spPr"))
    if sppr is None:
        sppr = etree.SubElement(cs, qn("c:spPr"))
        cs.insert(cs.index(cs.find(qn("c:chart"))) + 1, sppr)
    for e in list(sppr):
        sppr.remove(e)
    etree.SubElement(sppr, qn("a:noFill"))
    ln = etree.SubElement(sppr, qn("a:ln")); etree.SubElement(ln, qn("a:noFill"))
    return ch


def fill_series(s, color, line=None):
    if hasattr(s, "invert_if_negative"):
        s.invert_if_negative = False
    s.format.fill.solid()
    s.format.fill.fore_color.rgb = rgb(color)
    if line:
        s.format.line.color.rgb = rgb(line); s.format.line.width = Pt(0.75)
    else:
        s.format.line.fill.background()


def fill_point(s, i, color):
    s.invert_if_negative = False
    p = s.points[i]
    p.format.fill.solid()
    p.format.fill.fore_color.rgb = rgb(color)
    p.format.line.fill.background()
    dpt = p._ser.get_or_add_dPt_for_point(p._idx)
    if dpt.find(qn("c:invertIfNegative")) is None:
        e = etree.Element(qn("c:invertIfNegative")); e.set("val", "0")
        dpt.find(qn("c:idx")).addnext(e)


def err_bars(series, plus, minus, direction=None, color=INK, width=1.25):
    """Custom error bars from absolute half-widths. direction 'x' or 'y' for scatter; None for bar/line."""
    ser = series._element
    eb = etree.Element(qn("c:errBars"))
    if direction:
        etree.SubElement(eb, qn("c:errDir")).set("val", direction)
    etree.SubElement(eb, qn("c:errBarType")).set("val", "both")
    etree.SubElement(eb, qn("c:errValType")).set("val", "cust")
    etree.SubElement(eb, qn("c:noEndCap")).set("val", "0")
    for tag, vals in (("c:plus", plus), ("c:minus", minus)):
        t = etree.SubElement(eb, qn(tag))
        nl = etree.SubElement(t, qn("c:numLit"))
        etree.SubElement(nl, qn("c:formatCode")).text = "General"
        etree.SubElement(nl, qn("c:ptCount")).set("val", str(len(vals)))
        for i, v in enumerate(vals):
            pt = etree.SubElement(nl, qn("c:pt")); pt.set("idx", str(i))
            etree.SubElement(pt, qn("c:v")).text = repr(float(max(v, 0.0)))
    sp = etree.SubElement(eb, qn("c:spPr"))
    ln = etree.SubElement(sp, qn("a:ln")); ln.set("w", str(int(Pt(width))))
    sf = etree.SubElement(ln, qn("a:solidFill")); etree.SubElement(sf, qn("a:srgbClr")).set("val", color)
    # schema order: errBars sits before cat/val (bar, line) or xVal/yVal (scatter)
    anchor = None
    for t in ("c:cat", "c:val", "c:xVal", "c:yVal", "c:smooth", "c:shape", "c:extLst"):
        anchor = ser.find(qn(t))
        if anchor is not None:
            break
    anchor.addprevious(eb)


def data_labels(series, labels, size=TXT - 2, color=INK, pos=None):
    """Custom text label per point (editable rich text)."""
    for i, s in enumerate(labels):
        if s is None:
            continue
        dl = series.points[i].data_label
        tf = dl.text_frame
        tf.text = s
        for p in tf.paragraphs:
            for r in p.runs:
                r.font.name, r.font.size = FONT, Pt(size)
                r.font.color.rgb = rgb(color)
        if pos is not None:
            dl.position = pos


def marker(series, color, size=9, style=XL_MARKER_STYLE.CIRCLE, line=False, lw=2.0):
    series.marker.style = style
    series.marker.size = size
    series.marker.format.fill.solid()
    series.marker.format.fill.fore_color.rgb = rgb(color)
    series.marker.format.line.color.rgb = rgb("FFFFFF")
    series.marker.format.line.width = Pt(1)
    if line:
        series.format.line.color.rgb = rgb(color)
        series.format.line.width = Pt(lw)
    else:
        series.format.line.fill.background()
    series.smooth = False


def gap(chart, gap_width=60, overlap=0):
    pl = chart.plots[0]
    pl.gap_width = gap_width
    pl.overlap = overlap


# ------------------------------------------------------------------ figures
def fig_framework(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    cols = [("Field data", 0.45), ("Models", 3.55), ("Evaluation", 6.65), ("Outputs", 9.75)]
    for name, x in cols:
        box(s, x, 0.35, 2.9, 0.5, name, fill="F2F2F0", line=None, size=18, bold=True)
    for (_, x1), (_, x2) in zip(cols[:-1], cols[1:]):
        arrow(s, x1 + 2.95, 3.6, x2 - 0.05, 3.6, w=2.25)
    ds = FD["dataset"]
    dev = [v for v in ds.values() if v["split"] == "dev"]; lk = [v for v in ds.values() if v["split"] == "locked"]
    nd, nl = sum(v["files"] for v in dev), sum(v["files"] for v in lk)
    box(s, 0.45, 1.05, 2.9, 1.05, "Acoustic loggers\n4,096 Hz, 10 s each night", size=16, bold=True)
    box(s, 0.45, 2.3, 2.9, 1.05, "12 sites in 4 districts\n2,442 files, 61 deployments", size=16, bold=True)
    box(s, 0.45, 3.55, 2.9, 1.25, f"Development\nSites 1 to 9\n{nd:,} files, 51 deployments", fill=FILL_SOFT["GREY"], size=16, bold=True)
    box(s, 0.45, 5.0, 2.9, 1.25, f"Locked test\nSites 10 to 12\n{nl} files, 10 deployments", fill=FILL_SOFT["RED"], line=LEAK, size=16, bold=True)
    box(s, 3.55, 1.05, 2.9, 2.0, "Pretrained AST\nAudioSet weights, frozen\n86.2 M parameters\nlinear probe, 769 trained",
        fill=FILL_SOFT["AST"], line=AST, size=16, bold=True)
    box(s, 3.55, 3.25, 2.9, 1.35, "CNN from scratch\n4 blocks, 0.42 M trained", fill=FILL_SOFT["CNN"], line=CNN, size=16, bold=True)
    box(s, 3.55, 4.8, 2.9, 1.45, "LSTM from scratch\nBiLSTM on 20 MFCC\n0.14 M trained", fill=FILL_SOFT["LSTM"], line=LSTM, size=16, bold=True)
    box(s, 6.65, 1.05, 2.9, 1.55, "Leave one site out\nnested selection\nthreshold from training sites", size=16, bold=True)
    box(s, 6.65, 2.8, 2.9, 1.5, "Deployment bootstrap\n2,000 resamples\n95% intervals", size=16, bold=True)
    box(s, 6.65, 4.5, 2.9, 1.75, "One locked test\nfixed in advance\nscored once", fill=FILL_SOFT["RED"], line=LEAK, size=16, bold=True)
    for i, (t, b) in enumerate([("Detection", "leak or no leak per file"), ("Evidence", "which band carries it"),
                                ("Evidence tiers", "per deployment"), ("Cost", "time and memory")]):
        box(s, 9.75, 1.05 + i * 1.33, 2.9, 1.15, f"{t}\n{b}", fill="FFFFFF", size=16, bold=True)
    caption(s, n, "Research framework")


def fig_dataset(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    ds = FD["dataset"]
    sites = [str(k) for k in range(12, 0, -1)]
    cd = CategoryChartData()
    cd.categories = [f"Site {k}  ({ds[k]['deployments']})" for k in sites]
    cd.add_series("No leak", [ds[k]["no_leak"] for k in sites])
    cd.add_series("Leak", [ds[k]["leak"] for k in sites])
    ch = base_chart(s, XL_CHART_TYPE.BAR_STACKED, cd, 0.6, 0.35, 8.4, 6.3, legend=True)
    fill_series(ch.series[0], NOLEAK); fill_series(ch.series[1], LEAK)
    gap(ch, 45, 100)
    style_axis(ch.value_axis, "Files", grid=True, fmt="#,##0", minv=0, maxv=500, major=100)
    style_axis(ch.category_axis, None)
    # stacked bars need a 2 px surface gap
    for sr in ch.series:
        sr.format.line.color.rgb = rgb("FFFFFF"); sr.format.line.width = Pt(1.5)
    # side summary
    box(s, 9.4, 0.9, 3.4, 2.4, "Development\nSites 1 to 9\n2,159 files\n51 deployments\n556 leak files",
        fill=FILL_SOFT["GREY"], line=None, size=16, bold=True)
    box(s, 9.4, 3.6, 3.4, 2.4, "Locked test\nSites 10 to 12\n283 files\n10 deployments\n100 leak files",
        fill=FILL_SOFT["RED"], line=None, size=16, bold=True)
    text(s, 9.4, 6.15, 3.4, 0.5, "Number of deployments in brackets", size=14, color=INK2)
    caption(s, n, "Files per site and class")


def spectrogram_png():
    """Input-A log-mel of the leak example, rendered once as a picture (data raster)."""
    path = os.path.join(C.FIGURES, "spec_inputA_leak.png")
    if not os.path.exists(path):
        raise SystemExit("run f03_spectrogram.py in deep env  first")
    return path


def fig_pipeline(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    y0 = 0.5
    box(s, 0.35, y0 + 0.65, 1.7, 1.2, "10-s record\n4,096 Hz\n0 to 2,048 Hz", size=15, bold=True)
    arrow(s, 2.1, y0 + 1.25, 2.55, y0 + 1.25)
    box(s, 2.6, y0 + 0.65, 1.7, 1.2, "Resample\nto 16 kHz", size=15, bold=True)
    arrow(s, 4.35, y0 + 1.25, 4.75, y0 + 1.25)
    sp = spectrogram_png()
    s.shapes.add_picture(sp, Inches(4.8), Inches(y0 + 0.1), Inches(2.6), Inches(2.3))
    text(s, 4.8, y0 + 2.45, 2.6, 0.6, "128 log-mel bins\nshaded: above 2,048 Hz", size=14, color=INK2, align=PP_ALIGN.CENTER)
    arrow(s, 7.45, y0 + 1.25, 7.85, y0 + 1.25)
    box(s, 7.9, y0 + 0.2, 2.5, 2.1, "AST, frozen\n16 x 16 patches\n12 x 101 grid\n12 encoder layers", fill=FILL_SOFT["AST"],
        line=AST, size=15, bold=True)
    arrow(s, 10.45, y0 + 1.25, 10.85, y0 + 1.25)
    box(s, 10.9, y0 + 0.65, 2.1, 1.2, "Pooled\n768 values", size=15, bold=True)
    # second row: from embedding to tier
    y1 = 3.75
    arrow(s, 11.95, y0 + 1.9, 11.95, y1 - 0.05)
    box(s, 10.9, y1, 2.1, 1.25, "Linear probe\n769 weights\nscore p", fill=FILL_SOFT["AST"], line=AST, size=15, bold=True)
    arrow(s, 10.85, y1 + 0.62, 10.45, y1 + 0.62)
    box(s, 7.9, y1, 2.5, 1.25, "File decision\np >= threshold\nset on training sites", size=15, bold=True)
    arrow(s, 7.85, y1 + 0.62, 7.45, y1 + 0.62)
    box(s, 4.8, y1, 2.6, 1.25, "Night score\nmean p of the night\nS = median over nights", size=15, bold=True)
    arrow(s, 4.75, y1 + 0.62, 4.35, y1 + 0.62)
    box(s, 2.6, y1, 1.7, 1.25, "Persistence P\nshare of nights\nabove threshold", size=15, bold=True)
    arrow(s, 2.55, y1 + 0.62, 2.1, y1 + 0.62)
    # tiers ladder
    names = ["Major", "Significant", "Moderate", "Borderline", "Not flagged"]
    for i, nm in enumerate(names):
        box(s, 0.35, y1 - 0.65 + i * 0.52, 1.7, 0.45, nm, fill=TIER[4 - i], line=None, size=14,
            color=("FFFFFF" if i < 2 else INK), shape=MSO_SHAPE.RECTANGLE)
    text(s, 0.2, y1 + 1.95, 4.2, 0.9, "Tier cut points: quartiles of S over\nleaking training deployments", size=14, color=INK2)
    caption(s, n, "From a field record to a leak evidence tier")


def fig_signals(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    W = FD["waveforms"]
    names = {"Leak": "Leak", "No leak": "No leak", "Borderline": "Borderline"}
    ymax = max(max(abs(v) for v in w["env_max"] + w["env_min"]) for w in W)
    for i, w in enumerate(W):
        y = 0.3 + i * 2.15
        lab = "leak" if w["y"] == 1 else "no leak"
        panel(s, 0.3, y, f"({'abc'[i]})")
        text(s, 0.85, y, 7.5, 0.4, f"{names[w['example']]}: labelled {lab}, score {w['score']:.2f}, threshold {w['thr']:.2f}", size=15)
        cd = XyChartData()
        env = cd.add_series("waveform envelope")
        dt = w["t"][1] - w["t"][0]
        for t, a, b in zip(w["t"], w["env_max"], w["env_min"]):      # max and min of each 40 ms block, in turn
            env.add_data_point(t, a); env.add_data_point(t + dt / 2, b)
        ch = base_chart(s, XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, cd, 0.4, y + 0.35, 6.6, 1.8)
        for sr in ch.series:
            sr.format.line.color.rgb = rgb(AST); sr.format.line.width = Pt(0.75); sr.smooth = False
        lim = float(np.ceil((ymax + 0.02) * 10) / 10)
        style_axis(ch.value_axis, "Amplitude", size=14, minv=-lim, maxv=lim, major=(2 * lim if i == 2 else lim), fmt="0.0")
        style_axis(ch.category_axis, "Time (s)" if i == 2 else None, size=14, minv=0, maxv=10, major=2, fmt="0")
        x_at_bottom(ch)
        cd2 = XyChartData()
        ps = cd2.add_series("PSD")
        for f, p in zip(w["f"], w["psd_db"]):
            ps.add_data_point(f, p)
        ch2 = base_chart(s, XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, cd2, 7.3, y + 0.35, 5.7, 1.8)
        ch2.series[0].format.line.color.rgb = rgb(AST); ch2.series[0].format.line.width = Pt(1.25); ch2.series[0].smooth = False
        style_axis(ch2.value_axis, "dB", size=14, minv=-110, maxv=-30, major=(80 if i == 2 else 40), fmt="0", grid=True)
        style_axis(ch2.category_axis, "Frequency (Hz)" if i == 2 else None, size=14, minv=0, maxv=2048, major=500, fmt="#,##0")
        x_at_bottom(ch2)
    caption(s, n, "Three development files, chosen by a fixed rule")


def ci(d, k="auc"):
    return d[k + "_hi"] - d[k], d[k] - d[k + "_lo"]


def gap_chart(slide, rows_, x, y, w, h, title="Gap in AUC", minv=-0.1, maxv=0.35, major=0.1, colors=None):
    """Horizontal bars from zero with 95% CI; the label of each row carries the numbers."""
    cd = CategoryChartData()
    cd.categories = [f"{nm}\n{p['d']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]" for nm, p in rows_][::-1]
    cd.add_series("gap", [p["d"] for _, p in rows_][::-1])
    ch = base_chart(slide, XL_CHART_TYPE.BAR_CLUSTERED, cd, x, y, w, h)
    sr = ch.series[0]
    cols = (colors or [AST] * len(rows_))[::-1]
    for i, c in enumerate(cols):
        fill_point(sr, i, c)
    gap(ch, 80)
    err_bars(sr, [p["hi"] - p["d"] for _, p in rows_][::-1], [p["d"] - p["lo"] for _, p in rows_][::-1])
    style_axis(ch.value_axis, title, grid=True, minv=minv, maxv=maxv, major=major, fmt="0.0#")
    style_axis(ch.category_axis, None, size=14)
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    return ch


def P_(a, b):
    p = next(q for q in FD["paired"] if q["ref"] == a and q["other"] == b)
    return dict(d=p["d_auc"], lo=p["d_auc_lo"], hi=p["d_auc_hi"])


def fig_main(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    L_ = FD["loco"]
    tags = [("ast_A_pool_LP", "AST"), ("scratch_CNN", "CNN"), ("scratch_LSTM", "LSTM")]
    for j, (k, title) in enumerate([("auc", "AUC"), ("balanced_accuracy", "Balanced accuracy")]):
        panel(s, 0.3 + j * 3.6, 0.25, f"({'ab'[j]})")
        cd = CategoryChartData()
        cd.categories = [f"{nm}\n{L_[t][k]:.3f}" for t, nm in tags]
        cd.add_series(title, [L_[t][k] for t, _ in tags])
        ch = base_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, cd, 0.4 + j * 3.6, 0.6, 3.5, 6.0)
        sr = ch.series[0]
        for i, (_, nm) in enumerate(tags):
            fill_point(sr, i, MODEL_COL[nm])
        gap(ch, 70)
        style_axis(ch.value_axis, title + ", leave one site out", grid=True, minv=0.5, maxv=1.0, major=0.1, fmt="0.0")
        style_axis(ch.category_axis)
        err_bars(sr, [ci(L_[t], k)[0] for t, _ in tags], [ci(L_[t], k)[1] for t, _ in tags])
    panel(s, 7.75, 0.25, "(c)")
    gap_chart(s, [("AST minus CNN", P_("ast_A_pool_LP", "scratch_CNN")), ("AST minus LSTM", P_("ast_A_pool_LP", "scratch_LSTM")),
                  ("CNN minus LSTM", P_("scratch_CNN", "scratch_LSTM"))], 7.8, 0.6, 5.3, 5.2, colors=[AST, AST, CNN])
    text(s, 7.9, 5.9, 5.2, 0.8, "Paired by file. 95% intervals from 2,000\ndeployment resamples.", size=14, color=INK2)
    caption(s, n, "Leave-one-site-out results for the three deep models")


def fig_sites(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    P = FD["per_site"]
    sites = [k for k in range(1, 10) if P["AST"][str(k)] is not None]
    cd = XyChartData()
    for nm, off in (("AST", -0.18), ("CNN", 0.0), ("LSTM", 0.18)):
        sr = cd.add_series(nm)
        for k in sites:
            sr.add_data_point(k + off, P[nm][str(k)])
    ch = base_chart(s, XL_CHART_TYPE.XY_SCATTER, cd, 0.5, 0.35, 12.3, 6.1, legend=True)
    for sr, nm, st in zip(ch.series, ("AST", "CNN", "LSTM"), (XL_MARKER_STYLE.CIRCLE, XL_MARKER_STYLE.SQUARE, XL_MARKER_STYLE.TRIANGLE)):
        marker(sr, MODEL_COL[nm], size=14, style=st)
    style_axis(ch.value_axis, "AUC on the held-out site", grid=True, minv=0.2, maxv=1.0, major=0.2, fmt="0.0")
    style_axis(ch.category_axis, "Held-out development site", minv=0, maxv=10, major=1, fmt='[<0.5]"";[>9.5]"";0')
    x_at_bottom(ch)
    text(s, 0.9, 6.45, 11.5, 0.4, "Site 7 has no leak files, so its AUC is not defined.", size=14, color=INK2)
    caption(s, n, "AUC at each held-out development site")


def x_at_bottom(ch):
    ch.value_axis.crosses = XL_AXIS_CROSSES.MINIMUM


def inside_labels(sr, labels, n, color="FFFFFF"):
    data_labels(sr, labels, size=14, color=color)
    for p in range(n):
        sr.points[p].data_label.position = XL_LABEL_POSITION.INSIDE_BASE


def xy_line(slide, series, x, y, w, h, legend=False):
    cd = XyChartData()
    for nm, xs, ys, _ in series:
        sr = cd.add_series(nm)
        for a, b in zip(xs, ys):
            sr.add_data_point(a, b)
    ch = base_chart(slide, XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, cd, x, y, w, h, legend=legend)
    for sr, (_, _, _, sty) in zip(ch.series, series):
        sr.format.line.color.rgb = rgb(sty.get("color", AST))
        sr.format.line.width = Pt(sty.get("w", 2.0))
        sr.smooth = False
        if sty.get("dash"):
            sr.format.line.dash_style = sty["dash"]
    return ch


def fig_frequency(prs, n):
    from pptx.enum.dml import MSO_LINE
    s = prs.slides.add_slide(prs.slide_layouts[6])
    pb = FD["perbin"]
    f = pb["centre_hz"]
    nyq = ("Nyquist", [2048, 2048], [0.3, 1.0], dict(color=INK2, w=1.25, dash=MSO_LINE.DASH))
    panel(s, 0.3, 0.25, "(a)")
    ch = xy_line(s, [("Per-bin AUC", f, pb["auc"], dict(color=AST, w=2.25)), nyq], 0.4, 0.55, 6.3, 3.0)
    style_axis(ch.value_axis, "AUC of one bin", size=14, grid=True, minv=0.3, maxv=1.0, major=0.1, fmt="0.0")
    style_axis(ch.category_axis, None, size=14, minv=0, maxv=8000, major=2000, fmt="#,##0")
    x_at_bottom(ch)
    text(s, 2.75, 0.5, 2.6, 0.35, "2,048 Hz, logger limit", size=13, color=INK2)
    panel(s, 0.3, 3.55, "(b)")
    nyq2 = ("Nyquist", [2048, 2048], [-16, 2], dict(color=INK2, w=1.25, dash=MSO_LINE.DASH))
    ch2 = xy_line(s, [("Leak", f, pb["mean_leak"], dict(color=LEAK, w=2.25)),
                      ("No leak", f, pb["mean_noleak"], dict(color=NOLEAK, w=2.25)), nyq2], 0.4, 3.85, 6.3, 2.9)
    style_axis(ch2.value_axis, "Mean log-mel", size=14, grid=True, minv=-16, maxv=2, major=4, fmt="0")
    style_axis(ch2.category_axis, "Mel bin centre (Hz)", size=14, minv=0, maxv=8000, major=2000, fmt="#,##0")
    x_at_bottom(ch2)
    text(s, 4.6, 4.05, 1.9, 0.35, "Leak files", size=14, color=LEAK, bold=True)
    text(s, 4.6, 4.4, 1.9, 0.35, "No-leak files", size=14, color=INK2, bold=True)
    # (c) integrated gradients: share of the class contrast
    panel(s, 7.0, 0.25, "(c)")
    ig = FD["ig_contrast"]
    regs = [("Below 2,048 Hz", "in_band"), ("Straddle bins", "straddle"), ("Above 2,048 Hz", "dead")]
    cd = CategoryChartData()
    cd.categories = [r + "\n" + format(ig[f"fraction_of_contrast_{k}"]["est"], "+.0%") for r, k in regs]
    cd.add_series("share", [ig[f"fraction_of_contrast_{k}"]["est"] for _, k in regs])
    ch3 = base_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, cd, 7.1, 0.55, 5.9, 4.1)
    sr = ch3.series[0]
    for i, c in enumerate([AST, "9DBFEA", "C9C8C3"]):
        fill_point(sr, i, c)
    gap(ch3, 70)
    err_bars(sr, [ig[f"fraction_of_contrast_{k}"]["hi"] - ig[f"fraction_of_contrast_{k}"]["est"] for _, k in regs],
             [ig[f"fraction_of_contrast_{k}"]["est"] - ig[f"fraction_of_contrast_{k}"]["lo"] for _, k in regs])
    style_axis(ch3.value_axis, "Share of leak minus no-leak logit", size=14, grid=True, minv=-0.4, maxv=1.4, major=0.2, fmt="0%")
    style_axis(ch3.category_axis, None, size=14)
    ch3.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    text(s, 7.1, 4.75, 5.9, 1.9, "Integrated gradients, 100 development files\n(50 leak, 50 no leak). Baseline: an empty\n"
         "spectrogram. Shares add up to 100%.", size=14, color=INK2)
    caption(s, n, "Where the leak evidence sits in frequency")


def fig_inputs(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    L_ = FD["loco"]
    rows_ = [("ast_A_pool_LP", "Standard input A"), ("ast_Afill_pool_LP", "Input A, band above\n2,048 Hz emptied"),
             ("ast_B_pool_LP", "Native-band input B"), ("ast_A_mean_dead_LP", "Tokens above\n2,048 Hz only"),
             ("ast_DEADSTAT_LP", "Raw statistics above\n2,048 Hz, no AST"), ("scratch_CNN", "CNN reference")]
    cd = CategoryChartData()
    cd.categories = [nm for _, nm in rows_][::-1]
    cd.add_series("AUC", [L_[t]["auc"] for t, _ in rows_][::-1])
    ch = base_chart(s, XL_CHART_TYPE.BAR_CLUSTERED, cd, 0.4, 0.35, 8.6, 6.3)
    sr = ch.series[0]
    cols = [AST, "5E9BE0", "8DB2E3", "C3D5EE", "D9D8D3", CNN][::-1]
    for i, c in enumerate(cols):
        fill_point(sr, i, c)
    gap(ch, 50)
    err_bars(sr, [ci(L_[t])[0] for t, _ in rows_][::-1], [ci(L_[t])[1] for t, _ in rows_][::-1])
    style_axis(ch.value_axis, "AUC, leave one site out", grid=True, minv=0.5, maxv=1.0, major=0.1, fmt="0.0")
    style_axis(ch.category_axis, None)
    inside_labels(sr, [f"{L_[t]['auc']:.3f}" for t, _ in rows_][::-1], len(rows_), color=INK)
    P = {(p["ref"], p["other"]): p for p in FD["paired"]}
    g1 = P[("ast_A_pool_LP", "ast_Afill_pool_LP")]; g2 = P[("ast_B_pool_LP", "scratch_CNN")]; g3 = P[("ast_Afill_pool_LP", "scratch_CNN")]
    box(s, 9.3, 0.9, 3.6, 5.3,
        "Gaps in AUC [95% CI]\n\n"
        f"Standard minus emptied\n{g1['d_auc']:+.3f} [{g1['d_auc_lo']:+.3f}, {g1['d_auc_hi']:+.3f}]\n\n"
        f"Native band minus CNN\n{g2['d_auc']:+.3f} [{g2['d_auc_lo']:+.3f}, {g2['d_auc_hi']:+.3f}]\n\n"
        f"Emptied minus CNN\n{g3['d_auc']:+.3f} [{g3['d_auc_lo']:+.3f}, {g3['d_auc_hi']:+.3f}]",
        fill=FILL_SOFT["GREY"], line=None, size=15, bold=True, align=PP_ALIGN.LEFT)
    caption(s, n, "AST input variants and the band above 2,048 Hz")


def fig_layers(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    Ly = FD["layers"]
    ks = [f"ast_A_layer{k}_LP" for k in range(1, 13)]
    cd = CategoryChartData()
    cd.categories = [str(k) for k in range(1, 13)]
    cd.add_series("AST layer", [Ly[k]["auc"] for k in ks])
    cd.add_series("CNN from scratch", [FD["loco"]["scratch_CNN"]["auc"]] * 12)
    ch = base_chart(s, XL_CHART_TYPE.LINE_MARKERS, cd, 0.5, 0.35, 12.3, 6.2, legend=True)
    marker(ch.series[0], AST, size=11, line=True, lw=2.25)
    err_bars(ch.series[0], [Ly[k]["auc_hi"] - Ly[k]["auc"] for k in ks], [Ly[k]["auc"] - Ly[k]["auc_lo"] for k in ks], color=AST)
    sr = ch.series[1]
    sr.marker.style = XL_MARKER_STYLE.NONE
    from pptx.enum.dml import MSO_LINE
    sr.format.line.color.rgb = rgb(CNN); sr.format.line.width = Pt(2); sr.format.line.dash_style = MSO_LINE.DASH
    style_axis(ch.value_axis, "AUC, leave one site out", grid=True, minv=0.6, maxv=1.0, major=0.1, fmt="0.0")
    style_axis(ch.category_axis, "Encoder layer probed (frozen AST, input A)")
    caption(s, n, "Linear probe on each AST layer")


def fig_tiers(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    T = FD["tiers_dev"]["deployments_by_tier"]
    names = ["Not flagged", "Borderline", "Moderate", "Significant", "Major"]
    panel(s, 0.3, 0.25, "(a)")
    cd = CategoryChartData()
    cd.categories = names
    cd.add_series("No leak", [T[t]["no_leak"] for t in names])
    cd.add_series("Leak", [T[t]["leak"] for t in names])
    ch = base_chart(s, XL_CHART_TYPE.COLUMN_STACKED, cd, 0.4, 0.55, 6.2, 5.6, legend=True)
    fill_series(ch.series[0], NOLEAK, line="FFFFFF"); fill_series(ch.series[1], LEAK, line="FFFFFF")
    for sr in ch.series:
        sr.format.line.width = Pt(1.5)
    gap(ch, 55, 100)
    style_axis(ch.value_axis, "Development deployments", grid=True, minv=0, maxv=32, major=8, fmt="0")
    style_axis(ch.category_axis, None)
    pers = FD["tiers_dev"]["persistence_by_tier"]
    text(s, 0.6, 6.15, 6.0, 0.5, "Median persistence:  " + "   ".join(f"{pers[t]['median']:.2f}" for t in names), size=14, color=INK2)
    # (b) locked deployments; reference lines are chart series, so they sit exactly on the axis
    panel(s, 6.9, 0.25, "(b)")
    D = sorted(FD["locked_tiers"], key=lambda d: d["S"])
    cut = FD["locked_cuts"]
    cd = XyChartData()
    lk = cd.add_series("Leak deployment"); nl = cd.add_series("No-leak deployment")
    for i, d in enumerate(D):
        (lk if d["y"] == 1 else nl).add_data_point(d["S"], i + 1)
    refs = [("Threshold", cut["threshold"]), ("c1", cut["c1"]), ("c2", cut["c2"]), ("c3", cut["c3"])]
    for nm, v in refs:
        r_ = cd.add_series(nm)
        r_.add_data_point(v, 0.4); r_.add_data_point(v, len(D) + 0.6)
    ch2 = base_chart(s, XL_CHART_TYPE.XY_SCATTER, cd, 7.0, 0.55, 6.0, 5.75, legend=True)
    ch2.legend.position = XL_LEGEND_POSITION.BOTTOM
    marker(ch2.series[0], LEAK, size=14); marker(ch2.series[1], NOLEAK, size=14, style=XL_MARKER_STYLE.SQUARE)
    for j, (nm, v) in enumerate(refs):
        sr = ch2.series[2 + j]
        sr.marker.style = XL_MARKER_STYLE.NONE
        sr.format.line.color.rgb = rgb(INK2 if nm == "Threshold" else "8DB2E3")
        sr.format.line.width = Pt(1.5); sr.format.line.dash_style = MSO_LINE.DASH
        sr.smooth = False
        if nm == "Threshold":
            data_labels(sr, [None, nm], size=13, color=INK2)
            sr.points[1].data_label.position = XL_LABEL_POSITION.LEFT
        else:
            data_labels(sr, [None, nm], size=13, color=INK2)
            sr.points[1].data_label.position = XL_LABEL_POSITION.ABOVE
    # hide reference series from the legend
    leg = ch2.legend._element
    for j in range(2, 2 + len(refs)):
        le = etree.SubElement(leg, qn("c:legendEntry"))
        etree.SubElement(le, qn("c:idx")).set("val", str(j))
        etree.SubElement(le, qn("c:delete")).set("val", "1")
        leg.insert(1, le)
    style_axis(ch2.category_axis, "Deployment score S, locked Sites 10 to 12", size=14, minv=0.2, maxv=0.85, major=0.1, fmt="0.0")
    style_axis(ch2.value_axis, None, minv=0.0, maxv=len(D) + 1.2, major=1, line=False)
    ch2.value_axis.tick_label_position = XL_TICK_LABEL_POSITION.NONE
    caption(s, n, "Leak evidence tiers for development and locked deployments")


def fig_locked(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    Lk = FD["locked"]
    tags = [("ast_A_pool_LP", "AST"), ("scratch_CNN", "CNN"), ("scratch_LSTM", "LSTM")]
    for j, (k, title) in enumerate([("auc", "AUC"), ("balanced_accuracy", "Balanced accuracy")]):
        panel(s, 0.3 + j * 4.2, 0.25, f"({'ab'[j]})")
        cd = CategoryChartData()
        cd.categories = [f"{nm}\n{Lk[t][k]:.3f}" for t, nm in tags]
        cd.add_series(title, [Lk[t][k] for t, _ in tags])
        ch = base_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, cd, 0.4 + j * 4.2, 0.55, 4.0, 6.0)
        sr = ch.series[0]
        for i, (_, nm) in enumerate(tags):
            fill_point(sr, i, MODEL_COL[nm])
        gap(ch, 70)
        err_bars(sr, [ci(Lk[t], k)[0] for t, _ in tags], [ci(Lk[t], k)[1] for t, _ in tags])
        style_axis(ch.value_axis, title + ", locked test", grid=True, minv=0.5, maxv=1.0, major=0.1, fmt="0.0")
        style_axis(ch.category_axis)
    # (c) threshold shift
    panel(s, 8.7, 0.25, "(c)")
    TS = FD["threshold_shift"]
    cd = CategoryChartData()
    cd.categories = [nm for _, nm in tags]
    cd.add_series("Fixed on training sites", [TS[t]["fixed"] for t, _ in tags])
    cd.add_series("Best on locked sites", [TS[t]["locked_youden"] for t, _ in tags])
    ch = base_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, cd, 8.8, 0.55, 4.3, 6.0, legend=True)
    fill_series(ch.series[0], "C9C8C3"); fill_series(ch.series[1], INK2)
    gap(ch, 60, -5)
    style_axis(ch.value_axis, "Score threshold", grid=True, minv=0, maxv=1.0, major=0.2, fmt="0.0")
    style_axis(ch.category_axis)
    caption(s, n, "Single test on the locked Sites 10 to 12")


# ------------------------------------------------------------------ supplementary
def figS_heads(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    L_ = FD["loco"]
    rows_ = [("ast_A_pool_LP", "Input A, pooled, linear"), ("ast_A_pool_MLP", "Input A, pooled, MLP"),
             ("ast_A_pool_DEEP", "Input A, pooled, deep head"), ("ast_A_mean_in_LP", "Input A, in-band tokens, linear"),
             ("ast_A_max_in_LP", "Input A, in-band max, linear"), ("ast_A_mean_all_LP", "Input A, all tokens, linear"),
             ("ast_A_mean_in_MLP", "Input A, in-band tokens, MLP"), ("ast_A_mean_in_DEEP", "Input A, in-band tokens, deep head"),
             ("ast_B_pool_LP", "Input B, pooled, linear"), ("ast_B_mean_in_LP", "Input B, all tokens, linear"),
             ("ast_B_pool_MLP", "Input B, pooled, MLP"), ("ast_B_pool_DEEP", "Input B, pooled, deep head")]
    rows_ = sorted(rows_, key=lambda r: L_[r[0]]["auc"])
    cd = CategoryChartData()
    cd.categories = [nm for _, nm in rows_]
    cd.add_series("AUC", [L_[t]["auc"] for t, _ in rows_])
    ch = base_chart(s, XL_CHART_TYPE.BAR_CLUSTERED, cd, 0.4, 0.3, 12.5, 6.4)
    sr = ch.series[0]
    for i, (t, _) in enumerate(rows_):
        fill_point(sr, i, AST if t.endswith("_LP") else ("8DB2E3" if t.endswith("_MLP") else "C3D5EE"))
    gap(ch, 45)
    err_bars(sr, [ci(L_[t])[0] for t, _ in rows_], [ci(L_[t])[1] for t, _ in rows_])
    style_axis(ch.value_axis, "AUC, leave one site out", grid=True, minv=0.5, maxv=1.0, major=0.1, fmt="0.0")
    style_axis(ch.category_axis, None, size=14)
    caption(s, n, "Every AST head and input view", supp=True)


def figS_effects(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    E = FD["effects"][::-1]
    lab = {"M_mfcc0_mean": "MFCC 0, mean (overall level)", "B_20_100_share": "Share of power, 20 to 100 Hz",
           "S_centroid_hz": "Spectral centroid", "B_800_1200_db": "Level, 800 to 1,200 Hz", "S_peak_hz": "Peak frequency",
           "M_mfcc2_mean": "MFCC 2, mean", "T_zcr": "Zero-crossing rate", "B_600_800_db": "Level, 600 to 800 Hz",
           "S_entropy": "Spectral entropy", "S_rolloff85_hz": "85% roll-off frequency",
           "E_frame_rms_modulation": "Level modulation, 50 ms frames", "B_100_200_share": "Share of power, 100 to 200 Hz",
           "B_1200_1600_db": "Level, 1,200 to 1,600 Hz", "M_mfcc4_mean": "MFCC 4, mean", "B_800_1200_share": "Share of power, 800 to 1,200 Hz"}
    cd = CategoryChartData()
    cd.categories = [f"{lab.get(e['feature'], e['feature'])}  ({e['sites_same_sign']}/{e['sites_both_classes']})" for e in E]
    cd.add_series("delta", [e["cliffs_delta"] for e in E])
    ch = base_chart(s, XL_CHART_TYPE.BAR_CLUSTERED, cd, 0.4, 0.2, 12.5, 6.2)
    sr = ch.series[0]
    for i, e in enumerate(E):
        fill_point(sr, i, LEAK if e["cliffs_delta"] > 0 else NOLEAK)
    gap(ch, 40)
    err_bars(sr, [e["hi"] - e["cliffs_delta"] for e in E], [e["cliffs_delta"] - e["lo"] for e in E])
    style_axis(ch.value_axis, "Cliff's delta, leak against no leak (development files)", grid=True, minv=-1, maxv=1, major=0.25, fmt="0.00")
    style_axis(ch.category_axis, None, size=14)
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    text(s, 0.6, 6.42, 12.0, 0.4, "Red: higher in leak files. Brackets: sites with the same sign / sites with both classes.",
         size=18, color=INK2)
    caption(s, n, "Physical features that separate leak from no-leak files", supp=True)


def figS_igbins(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    from pptx.enum.dml import MSO_LINE
    f = FD["perbin"]["centre_hz"]
    ig = FD["ig_perbin"]
    mx = max(max(ig["abs_leak"]), max(ig["abs_noleak"]))
    ch = xy_line(s, [("Leak files", f, ig["abs_leak"], dict(color=LEAK, w=2.0)),
                     ("No-leak files", f, ig["abs_noleak"], dict(color=NOLEAK, w=2.0)),
                     ("2,048 Hz", [2048, 2048], [0, mx * 1.05], dict(color=INK2, w=1.25, dash=MSO_LINE.DASH))],
                 0.4, 0.3, 12.5, 6.3, legend=True)
    style_axis(ch.value_axis, "Mean |integrated gradient| per bin", grid=True, minv=0, fmt="0.00")
    style_axis(ch.category_axis, "Mel bin centre (Hz)", minv=0, maxv=8000, major=1000, fmt="#,##0")
    caption(s, n, "Attribution per mel bin for leak and no-leak files", supp=True)


def figS_tones(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    T = FD["deadband"]["tones"]
    cd = CategoryChartData()
    cd.categories = [f"{t['f0']:,.0f} Hz tone\nimage at {t['image_hz']:,.0f} Hz" for t in T]
    cd.add_series("Image level", [t["image_db"] for t in T])
    ch = base_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, cd, 0.5, 0.3, 8.0, 6.3)
    fill_series(ch.series[0], AST)
    gap(ch, 80)
    style_axis(ch.value_axis, "Image level after resampling (dB re tone)", grid=True, minv=-100, maxv=0, major=20, fmt="0")
    style_axis(ch.category_axis, None, size=14)
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    data_labels(ch.series[0], [f"{t['image_db']:.1f} dB" for t in T], size=14)
    box(s, 8.9, 0.9, 4.0, 4.8, "Why the band above\n2,048 Hz is not empty\n\nThe 4,096 to 16,000 Hz\nresampler leaves a weak copy\n"
        "of each tone at 4,096 Hz\nminus its frequency.\n\nThe log-mel floor sits\nfar below these copies.",
        fill=FILL_SOFT["GREY"], line=None, size=15, bold=True, align=PP_ALIGN.LEFT)
    caption(s, n, "Resampling images of pure tones", supp=True)


def figS_protocols(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    a, b = FD["loco"]["ast_A_pool_LP"], FD["loro"]["ast_A_pool_LP"]
    cd = CategoryChartData()
    cd.categories = ["Leave one site out\n(9 folds)", "Leave one district out\n(4 folds)"]
    cd.add_series("AUC", [a["auc"], b["auc"]])
    ch = base_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, cd, 0.5, 0.3, 7.0, 6.3)
    fill_series(ch.series[0], AST)
    gap(ch, 90)
    err_bars(ch.series[0], [a["auc_hi"] - a["auc"], b["auc_hi"] - b["auc"]], [a["auc"] - a["auc_lo"], b["auc"] - b["auc_lo"]])
    style_axis(ch.value_axis, "AUC, AST (input A, linear probe)", grid=True, minv=0.5, maxv=1.0, major=0.1, fmt="0.0")
    style_axis(ch.category_axis, None)
    inside_labels(ch.series[0], [f"{a['auc']:.3f}", f"{b['auc']:.3f}"], 2)
    caption(s, n, "Holding out a whole district", supp=True)


def set_meta(prs, title):
    cp = prs.core_properties
    cp.author = "Ridwan Taiwo"
    cp.last_modified_by = "Ridwan Taiwo"
    cp.title = title
    cp.comments = ""
    cp.subject = ""
    cp.keywords = ""


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    main_figs = [fig_framework, fig_dataset, fig_pipeline, fig_signals, fig_main, fig_sites, fig_frequency, fig_inputs,
                 fig_layers, fig_tiers, fig_locked]
    for i, f in enumerate(main_figs, 1):
        f(prs, i)
    supp = [figS_heads, figS_protocols, figS_effects, figS_igbins, figS_tones]
    for i, f in enumerate(supp, 1):
        f(prs, i)
    set_meta(prs, "this study figures")
    prs.save(OUT)
    print("saved", OUT, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
