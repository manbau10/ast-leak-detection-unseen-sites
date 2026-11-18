"""Method figures (three slides). Shapes, text and native PowerPoint equations only (f07_eq.add_equation).

M1  adapting the frozen AST to 4,096 Hz logger data: two input front ends, the frozen encoder, our read-outs and heads
M2  the CNN and LSTM trained from scratch, and the shared training rule
M3  nested site hold-out, the threshold, the deployment score and tiers, the bootstrap, and integrated gradients
Every setting drawn here was checked against s4, s6, s7, s11, s12, s13 and nested.py on 6 Nov 2026.
All text is 18 pt or larger (writing skill, 11 Nov 2026). Settings dropped from the figures to make room
(window lengths in ms, dropout, weight decay, token views, head sizes) are stated in the manuscript text instead.
Equations shown in these figures (do not repeat them in the manuscript text):
  mel scale; AST normalisation; encoder residual block; pooled read-out; layer read-out; linear probe;
  class-weighted BCE and its weight; MFCC (DCT-II); Youden threshold; AUC gap; night score, deployment score S,
  persistence P; tier rule; integrated gradients and its step points.
"""
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.enum.dml import MSO_LINE

from f02_figdeck import text, box, arrow, caption, INK, INK2, AST, CNN, LSTM, LEAK, NOLEAK, TIER, FILL_SOFT, FD
from f06_figdeck_v2 import container, shape, line, padlock, picture, nodes, connect, block_arrow, GREEN, ORANGE, BLUE, PURPLE, RED
from f07_eq import add_equation

LG = "13865C"   # darker aqua for LSTM text
T = 18


def eq(s, latex, x, y, w=4.0, h=0.55, size=T, align="left"):
    return add_equation(s, latex, x, y, w, h, size=size, align=align)


# ------------------------------------------------------------------ M1
def fig_m_ast(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    # (a) front ends
    container(s, 0.1, 0.1, 4.35, 6.6, "(a)  Two front ends", ORANGE, fill="FFFAF4")
    box(s, 0.25, 0.55, 4.05, 0.5, "Logger record: 4,096 Hz, 10 s", fill="FFFFFF", line=INK2, size=T, bold=True)
    arrow(s, 1.22, 1.07, 1.22, 1.27); arrow(s, 3.27, 1.07, 3.27, 1.27)
    text(s, 0.25, 1.22, 1.95, 0.38, "Input A", size=T, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
    text(s, 2.3, 1.22, 1.95, 0.38, "Input B", size=T, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
    for i, (a, b, h) in enumerate([("Resample\nto 16 kHz", "No\nresampling", 0.72), ("25 ms window\n20 to 8,000 Hz", "100 ms window\n20 to 2,048 Hz", 0.72),
                                   ("998 frames", "1,014 frames", 0.45)]):
        y = [1.62, 2.42, 3.22][i]
        box(s, 0.25, y, 1.95, h, a, fill="FFFFFF", line=ORANGE, size=T)
        box(s, 2.3, y, 1.95, h, b, fill="FFFFFF", line=ORANGE, size=T)
    eq(s, r"m(f)=1127\,\ln\left(1+\frac{f}{700}\right)", 0.25, 3.78, 4.05, 0.65)
    eq(s, r"\tilde{X}=\frac{X-\mu}{2\sigma}", 0.25, 4.45, 1.9, 0.7)
    text(s, 2.15, 4.45, 2.25, 0.7, "A: AudioSet μ, σ\nB: training sites", size=T, color=INK2)
    picture(s, "thumb_A_leak.png", 0.25, 5.25, 1.95, 0.9)
    picture(s, "thumb_B_leak.png", 2.3, 5.25, 1.95, 0.9)
    text(s, 0.2, 6.17, 2.05, 0.38, "Top band empty", size=T, color=INK2, align=PP_ALIGN.CENTER)
    text(s, 2.25, 6.17, 2.05, 0.38, "All bins in band", size=T, color=INK2, align=PP_ALIGN.CENTER)
    block_arrow(s, 4.47, 3.2, 0.25, 0.4)
    # (b) frozen encoder
    container(s, 4.75, 0.1, 4.25, 6.6, "(b)  Frozen AST encoder", BLUE, fill="F6F9FD")
    padlock(s, 8.6, 0.17, 0.26, color=BLUE)
    text(s, 4.9, 0.5, 4.0, 0.7, "16 x 16 patches, stride 10:\n12 x 101 = 1,212 patches", size=T)
    rows = [("F2C1A6", 6), ("FBE3D5", 1), ("D9D8D3", 5)]   # in band, straddle, above 2,048 Hz (drawn top = high frequency)
    yy = 1.3
    for col, k in rows[::-1]:
        for r_ in range(k):
            for c in range(14):
                shape(s, MSO_SHAPE.RECTANGLE, 4.95 + c * 0.155, yy, 0.14, 0.085, fill=col)
            yy += 0.095
    text(s, 7.2, 1.22, 1.75, 0.38, "> 2,048 Hz", size=T, color=INK2)
    text(s, 7.2, 1.6, 1.75, 0.38, "edge", size=T, color=INK2)
    text(s, 7.2, 1.98, 1.75, 0.38, "< 2,048 Hz", size=T, color=INK2)
    arrow(s, 6.0, 2.46, 6.0, 2.6)
    box(s, 4.95, 2.62, 3.9, 0.75, "Patch embedding\n[CLS], [DIST], 1,212 tokens", fill="FFFFFF", line=BLUE, size=T, bold=True)
    arrow(s, 6.0, 3.39, 6.0, 3.52)
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 4.95, 3.54, 3.9, 2.1, fill="EAF2FB", line=BLUE, lw=1.25)
    text(s, 5.05, 3.57, 3.7, 0.38, "Block, repeated 12 times", size=T, bold=True, color=BLUE)
    eq(s, r"\mathbf{z}'_{\ell}=\mathrm{MSA}\left(\mathrm{LN}(\mathbf{z}_{\ell-1})\right)+\mathbf{z}_{\ell-1}", 5.0, 3.97, 3.8, 0.5)
    eq(s, r"\mathbf{z}_{\ell}=\mathrm{MLP}\left(\mathrm{LN}(\mathbf{z}'_{\ell})\right)+\mathbf{z}'_{\ell}", 5.0, 4.45, 3.8, 0.5)
    text(s, 5.05, 4.92, 3.75, 0.7, "MSA: 12-head attention\nMLP: 3,072 hidden units", size=T, color=INK2)
    box(s, 4.95, 5.74, 3.9, 0.86, "AudioSet weights, frozen\n86.2 M weights, no fine-tuning", fill=FILL_SOFT["AST"], line=None, size=T, bold=True)
    block_arrow(s, 9.02, 3.2, 0.25, 0.4)
    # (c) read-outs and heads
    container(s, 9.3, 0.1, 3.93, 6.6, "(c)  Read-outs and heads", PURPLE, fill="F8F7FD")
    text(s, 9.45, 0.5, 3.7, 0.38, "Pooled output", size=T, bold=True)
    eq(s, r"\mathbf{h}=\tfrac{1}{2}\left(\mathrm{LN}(\mathbf{z}_{12})^{\mathrm{cls}}+\mathrm{LN}(\mathbf{z}_{12})^{\mathrm{dist}}\right)", 9.38, 0.88, 3.85, 0.55)
    text(s, 9.45, 1.45, 3.7, 0.7, "Layer probe: same\npooling at layer k", size=T, color=INK2)
    eq(s, r"\mathbf{h}^{(k)}=\tfrac{1}{2}\left(\mathbf{z}_{k}^{\mathrm{cls}}+\mathbf{z}_{k}^{\mathrm{dist}}\right)", 9.45, 2.15, 3.7, 0.55)
    text(s, 9.45, 2.78, 3.7, 0.7, "Control: bins 70 to 127\nset to the log floor", size=T, color=INK2)
    line(s, 9.45, 3.58, 13.1, 3.58, color="D9D8D3", dash=MSO_LINE.DASH)
    text(s, 9.45, 3.64, 3.7, 0.38, "Linear probe, 769 weights", size=T, bold=True)
    eq(s, r"p=\sigma\left(\mathbf{w}^{\top}\frac{\mathbf{h}-\boldsymbol{\mu}}{\mathbf{s}}+b\right)", 9.45, 4.02, 3.7, 0.7)
    text(s, 9.45, 4.74, 3.7, 0.38, "C tuned on training sites", size=T, color=INK2)
    text(s, 9.45, 5.2, 3.7, 0.38, "Other heads tested", size=T, bold=True)
    a = nodes(s, 9.6, 5.68, 4, 0.22, r=0.12, fill="FFFFFF", color=PURPLE)
    b = nodes(s, 10.0, 5.79, 3, 0.22, r=0.12, fill="D6D2F2", color=PURPLE)
    c = nodes(s, 10.4, 6.01, 1, 0.22, r=0.12, fill=PURPLE, color=PURPLE)
    connect(s, a, b); connect(s, b, c)
    text(s, 10.7, 5.66, 2.45, 0.75, "MLP, deep head\n(Figure S1)", size=T, color=INK2)
    caption(s, n, "How the pretrained AST was adapted to the logger data")


# ------------------------------------------------------------------ M2
def cuboid(s, x, y, w, h, d, fill, edge):
    shape(s, MSO_SHAPE.CUBE, x, y, w + d, h + d, fill=fill, line=edge, lw=1)


def fig_m_scratch(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    # (a) CNN
    container(s, 0.1, 0.1, 13.13, 2.95, "(a)  CNN from scratch, 421,825 weights", CNN, fill="FFFAF6")
    box(s, 0.3, 0.7, 1.6, 1.15, "Input B\n253 x 128", fill="FFFFFF", line=CNN, size=T)
    shapes_ = [("1", 0.95, 1.6), ("32", 0.8, 1.35), ("64", 0.62, 1.1), ("128", 0.48, 0.85), ("256", 0.36, 0.62)]
    x = 2.15
    for i, (lab, w, h) in enumerate(shapes_):
        cuboid(s, x, 0.62 + (1.6 - h) / 2, w * 0.45, h * 0.8, 0.18 + i * 0.03, "FBD8C5" if i else "FFFFFF", CNN)
        text(s, x - 0.2, 2.0, 0.95, 0.38, lab, size=T, color=INK2, align=PP_ALIGN.CENTER)
        if i < 4:
            arrow(s, x + w * 0.45 + 0.25, 1.3, x + w * 0.45 + 0.55, 1.3)
        x += w * 0.45 + 0.65
    box(s, 6.95, 0.85, 1.35, 0.9, "Global\npool", fill="FFFFFF", line=CNN, size=T)
    arrow(s, 8.32, 1.3, 8.5, 1.3)
    box(s, 8.52, 0.85, 1.45, 0.9, "Dense\n128, 1", fill="FFFFFF", line=CNN, size=T)
    arrow(s, 9.99, 1.3, 10.17, 1.3)
    box(s, 10.2, 0.95, 0.8, 0.7, "p", fill=FILL_SOFT["CNN"], line=CNN, size=20, bold=True)
    text(s, 11.2, 0.75, 1.95, 1.1, "Dropout 0.3\nNo pretraining", size=T, color=INK2)
    text(s, 0.3, 2.47, 1.7, 0.38, "Each arrow:", size=T, bold=True)
    for i, t_ in enumerate(["Conv 3 x 3", "BatchNorm", "ReLU", "MaxPool 2"]):
        box(s, 2.05 + i * 1.8, 2.45, 1.6, 0.44, t_, fill="FFFFFF", line="F2B9A0", size=T)
        if i < 3:
            arrow(s, 2.05 + i * 1.8 + 1.6, 2.67, 2.05 + (i + 1) * 1.8, 2.67, w=1.0)
    text(s, 9.4, 2.47, 3.7, 0.38, "Numbers: channels", size=T, color=INK2)
    # (b) LSTM
    container(s, 0.1, 3.15, 6.5, 3.55, "(b)  LSTM from scratch, 143,489 weights", LG, fill="F5FBF8")
    box(s, 0.3, 3.62, 1.6, 0.75, "Input B\n253 x 128", fill="FFFFFF", line=LSTM, size=T)
    arrow(s, 1.92, 4.0, 2.12, 4.0)
    box(s, 2.14, 3.62, 1.7, 0.75, "DCT-II\nk = 0 to 19", fill="FFFFFF", line=LSTM, size=T)
    text(s, 4.0, 3.62, 2.5, 0.75, "20 MFCC per\nframe", size=T, color=INK2)
    eq(s, r"c_k=\alpha_k\sum_{m=0}^{127}X_m\cos\left(\frac{\pi k(2m+1)}{256}\right)", 0.3, 4.42, 6.2, 0.75, align="center")
    for layer in range(2):
        y = 5.2 + layer * 0.5
        xs = [0.5 + i * 0.78 for i in range(5)]
        for i, x in enumerate(xs):
            shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, 0.5, 0.36, fill="D4F0E5", line=LSTM, lw=1)
            if i < 4:
                arrow(s, x + 0.5, y + 0.11, x + 0.78, y + 0.11, color=LG, w=1.25)
                arrow(s, x + 0.78, y + 0.25, x + 0.5, y + 0.25, color="7FC9A9", w=1.25)
        text(s, 4.5, y - 0.01, 2.0, 0.38, f"BiLSTM {layer + 1}", size=T, color=INK2)
    box(s, 0.4, 6.2, 2.5, 0.42, "Mean over time", fill="FFFFFF", line=LSTM, size=T)
    arrow(s, 2.92, 6.41, 3.15, 6.41)
    box(s, 3.17, 6.2, 1.4, 0.42, "Dense: p", fill=FILL_SOFT["LSTM"], line=LSTM, size=T, bold=True)
    text(s, 4.7, 6.22, 1.85, 0.38, "64 per side", size=T, color=INK2)
    # (c) shared training
    container(s, 6.75, 3.15, 6.48, 3.55, "(c)  Training rule for CNN and LSTM", INK2, fill="FAFAF8")
    text(s, 6.9, 3.58, 6.2, 0.38, "Class-weighted cross-entropy", size=T, bold=True)
    eq(s, r"\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}\left[w_{+}\,y_i\ln p_i+(1-y_i)\ln(1-p_i)\right]", 6.85, 3.98, 6.35, 0.75)
    eq(s, r"w_{+}=\frac{N_0}{N_1}", 6.9, 4.72, 1.9, 0.7)
    text(s, 8.85, 4.75, 4.3, 0.7, "N0, N1: no-leak and\nleak training files", size=T, color=INK2)
    for i, t_ in enumerate(["AdamW\n0.001", "30 epochs\nbatch 32", "Train-site\nscaling", "5 seeds\naveraged"]):
        box(s, 6.9 + i * 1.57, 5.48, 1.52, 1.1, t_, fill="FFFFFF", line=INK2, size=T, bold=True)
    caption(s, n, "Deep models trained from scratch and their shared training rule")


# ------------------------------------------------------------------ M3
def fig_m_eval(prs, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    # (a) nested hold-out
    container(s, 0.1, 0.1, 6.5, 3.55, "(a)  Nested leave-one-site-out", PURPLE, fill="F8F7FD")
    for i in range(9):
        test = i == 3
        box(s, 0.3 + i * 0.68, 0.58, 0.6, 0.45, f"S{i + 1}", fill=(ORANGE if test else "D6D2F2"), line=None, size=T, bold=test,
            color=("FFFFFF" if test else INK))
    text(s, 0.3, 1.08, 6.2, 0.38, "Site k scored once; 8 sites train", size=T, color=INK2)
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.3, 1.52, 3.3, 2.03, fill="FFFFFF", line=PURPLE, lw=1.25, dash=MSO_LINE.DASH)
    text(s, 0.4, 1.55, 3.1, 0.38, "Inner loop, 8 sites", size=T, bold=True, color=PURPLE)
    for r_ in range(4):
        for i in range(8):
            shape(s, MSO_SHAPE.RECTANGLE, 0.45 + i * 0.24, 2.0 + r_ * 0.2, 0.21, 0.16, fill=(ORANGE if i == r_ else "D6D2F2"))
    text(s, 2.42, 1.95, 1.15, 0.75, "8 folds\n(4 shown)", size=T, color=INK2)
    text(s, 0.4, 2.82, 3.15, 0.7, "Choose setting\nand threshold", size=T)
    eq(s, r"\tau=\mathrm{arg\,max}_{t}\,J(t)", 3.65, 1.52, 2.9, 0.5)
    eq(s, r"J=\mathrm{TPR}-\mathrm{FPR}", 3.65, 2.0, 2.9, 0.5)
    text(s, 3.7, 2.55, 2.85, 1.0, "Refit, score Site k\nwith τ fixed. Repeat\nfor all 9 sites.", size=T)
    # (b) bootstrap
    container(s, 0.1, 3.75, 6.5, 2.95, "(b)  Deployment bootstrap", PURPLE, fill="F8F7FD")
    for i in range(3):
        for j in range(6):
            leak = (i * 6 + j) % 4 == 0
            shape(s, MSO_SHAPE.OVAL, 0.35 + j * 0.3, 4.25 + i * 0.3, 0.24, 0.24, fill=(LEAK if leak else NOLEAK))
    shape(s, MSO_SHAPE.CIRCULAR_ARROW, 2.25, 4.3, 0.6, 0.6, fill=PURPLE)
    text(s, 0.3, 5.25, 2.85, 1.15, "Resample\ndeployments\n2,000 times", size=T)
    eq(s, r"\Delta=\mathrm{AUC}_{\mathrm{AST}}-\mathrm{AUC}_{\mathrm{other}}", 3.2, 4.2, 3.35, 0.55)
    text(s, 3.25, 4.85, 3.25, 1.4, "Same files in each\nresample; 2.5th and\n97.5th percentiles", size=T)
    # (c) decision layer
    container(s, 6.75, 0.1, 6.48, 3.55, "(c)  From file scores to a tier", RED, fill="FDF7F7")
    for d in range(3):                                     # three nights of files
        for f in range(4):
            v = [0.72, 0.65, 0.8, 0.7, 0.4, 0.58, 0.62, 0.55, 0.81, 0.77, 0.69, 0.74][d * 4 + f]
            c = "%02X%02X%02X" % tuple(int(a + (b - a) * v) for a, b in zip((0xEE, 0xED, 0xEA), (0xB2, 0x1F, 0x24)))
            shape(s, MSO_SHAPE.RECTANGLE, 6.95 + f * 0.26, 0.6 + d * 0.34, 0.22, 0.26, fill=c)
        text(s, 8.02, 0.55 + d * 0.34, 1.0, 0.36, f"night {d + 1}", size=T, color=INK2)
    eq(s, r"\bar{p}_n=\frac{1}{|F_n|}\sum_{i\in F_n}p_i", 9.05, 0.48, 2.05, 0.75)
    eq(s, r"S=\mathrm{median}_{n}\,\bar{p}_n", 11.2, 0.55, 2.0, 0.55)
    eq(s, r"P=\frac{1}{N}\sum_{n=1}^{N}\mathbb{1}\left[\bar{p}_n\geq\tau\right]", 9.05, 1.2, 2.75, 0.75)
    text(s, 11.75, 1.3, 1.45, 0.38, "persistence", size=T, color=INK2)
    names = ["Not flagged", "Borderline", "Moderate", "Significant", "Major"]
    rules = [r"S<\tau", r"\tau\leq S<q_1", r"q_1\leq S<q_2", r"q_2\leq S<q_3", r"S\geq q_3"]
    for i, (nm, rl) in enumerate(zip(names, rules)):
        y = 1.98 + i * 0.32
        shape(s, MSO_SHAPE.RECTANGLE, 6.95, y, 1.5, 0.3, fill=TIER[i])
        text(s, 6.9, y - 0.04, 1.6, 0.36, nm, size=T, color=("FFFFFF" if i >= 3 else INK), align=PP_ALIGN.CENTER)
        eq(s, rl, 8.55, y - 0.09, 1.95, 0.45)
    text(s, 10.55, 2.0, 2.65, 1.55, "q1 to q3: quartiles\nof S over leaking\ntraining deployments", size=T, color=INK2)
    # (d) attribution
    container(s, 6.75, 3.75, 6.48, 2.95, "(d)  Integrated gradients per mel bin", RED, fill="FDF7F7")
    eq(s, r"\mathrm{IG}_{b}=\sum_{t}(x_{bt}-x'_{bt})\frac{1}{K}\sum_{k=1}^{K}\frac{\partial F\left(\mathbf{x}'+\alpha_k(\mathbf{x}-\mathbf{x}')\right)}{\partial x_{bt}}",
       6.85, 4.18, 6.35, 0.9)
    eq(s, r"\alpha_k=\frac{k-0.5}{K}", 6.9, 5.08, 2.2, 0.7)
    text(s, 9.2, 5.15, 3.95, 0.38, "K = 32; x′: empty record", size=T, color=INK2)
    text(s, 6.9, 5.82, 6.25, 0.75, "Summed below 2,048 Hz, at the edge\nand above it; 100 development files", size=T, color=INK2)
    caption(s, n, "Evaluation, decision and attribution")
