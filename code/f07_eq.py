"""Editable PowerPoint equations from LaTeX.

LaTeX -> MathML (latex2mathml) -> OMML (Office's own MML2OMML.XSL) -> an a14:m paragraph inside a text
box, wrapped in mc:AlternateContent so PowerPoint shows a native, editable equation. The fallback is the
same text box with a plain-text version of the equation, for readers that do not support a14.
"""
import copy
from lxml import etree
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import latex2mathml.converter as l2m

XSL = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"
_xslt = etree.XSLT(etree.parse(XSL))
NS = dict(m="http://schemas.openxmlformats.org/officeDocument/2006/math",
          a="http://schemas.openxmlformats.org/drawingml/2006/main",
          a14="http://schemas.microsoft.com/office/drawing/2010/main",
          mc="http://schemas.openxmlformats.org/markup-compatibility/2006",
          p="http://schemas.openxmlformats.org/presentationml/2006/main",
          w="http://schemas.openxmlformats.org/wordprocessingml/2006/main")


def omml(latex):
    mml = etree.fromstring(l2m.convert(latex).encode())
    om = _xslt(mml).getroot()                     # m:oMath
    for el in list(om.iter()):                    # drop Word run properties; PowerPoint uses a:rPr
        if isinstance(el.tag, str) and el.tag.startswith("{%s}" % NS["w"]):
            el.getparent().remove(el)
    return om


def _rpr(size, color, bold=False):
    r = etree.Element("{%s}rPr" % NS["a"])
    r.set("lang", "en-GB"); r.set("sz", str(int(size * 100))); r.set("i", "1")
    if bold:
        r.set("b", "1")
    sf = etree.SubElement(r, "{%s}solidFill" % NS["a"])
    etree.SubElement(sf, "{%s}srgbClr" % NS["a"]).set("val", color)
    etree.SubElement(r, "{%s}latin" % NS["a"]).set("typeface", "Cambria Math")
    etree.SubElement(r, "{%s}cs" % NS["a"]).set("typeface", "Cambria Math")
    return r


def add_equation(slide, latex, x, y, w, h, size=16, color="0B0B0B", align="left", fallback=None):
    """Adds one editable equation (one line) as a text box. Returns the text-box shape (fallback part)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = fallback or latex
    r.font.size = Pt(size); r.font.name = "Cambria Math"; r.font.color.rgb = RGBColor.from_string(color)
    sp = tb._element
    choice_sp = copy.deepcopy(sp)
    ap = choice_sp.find(".//{%s}p" % NS["a"])
    for ch in list(ap):
        if ch.tag != "{%s}pPr" % NS["a"]:
            ap.remove(ch)
    ppr = ap.find("{%s}pPr" % NS["a"])
    if ppr is None:
        ppr = etree.SubElement(ap, "{%s}pPr" % NS["a"]); ap.insert(0, ppr)
    ppr.set("algn", {"left": "l", "center": "ctr", "right": "r"}[align])
    m = etree.SubElement(ap, "{%s}m" % NS["a14"])
    para = etree.SubElement(m, "{%s}oMathPara" % NS["m"])
    pp = etree.SubElement(para, "{%s}oMathParaPr" % NS["m"])
    etree.SubElement(pp, "{%s}jc" % NS["m"]).set("{%s}val" % NS["m"], {"left": "left", "center": "centerGroup", "right": "right"}[align])
    om = omml(latex)
    for mr in om.iter("{%s}r" % NS["m"]):
        old = mr.find("{%s}rPr" % NS["a"])
        if old is not None:
            mr.remove(old)
        t = mr.find("{%s}t" % NS["m"])
        t.addprevious(_rpr(size, color))
    para.append(om)
    end = etree.SubElement(ap, "{%s}endParaRPr" % NS["a"]); end.set("lang", "en-GB"); end.set("sz", str(int(size * 100)))
    alt = etree.Element("{%s}AlternateContent" % NS["mc"], nsmap={"mc": NS["mc"]})
    choice = etree.SubElement(alt, "{%s}Choice" % NS["mc"], nsmap={"a14": NS["a14"]})
    choice.set("Requires", "a14")
    choice.append(choice_sp)
    fb = etree.SubElement(alt, "{%s}Fallback" % NS["mc"])
    sp.addprevious(alt)
    fb.append(sp)
    return tb
