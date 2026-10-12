"""
generate_paper.py — Builds the complete camera-ready IEEE conference paper in DOCX and exports to PDF.
Paper: ViT-XAI-Lyo: An Attention-Explainable Vision Transformer Framework for Lyophilized Cake
       Defect Classification with Regulatory-Aligned Interpretability
Conference: ICRM 2026 (Paper #1239)
Authors: Mathan M, Chandru S, Murugayesu A, Dr. Shanmuga sundar G
"""

import os
import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_OUT = Path("ViT-XAI-Lyo_Final_Paper.docx")
PDF_OUT = Path("ViT-XAI-Lyo_Final_Paper.pdf")
ARTIFACTS_DIR = Path(".artifacts/submission-readiness")
FIG_DIR = Path("results/figures")

def set_cell_margins(cell, top=50, bottom=50, left=80, right=80):
    """Set inner margins for a table cell in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex="F2F2F2"):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def set_cell_border(cell, **kwargs):
    """
    kwargs can be top, bottom, left, right.
    val: 'single', 'double', 'none', etc.
    sz: size in 1/8 pt (e.g. 4=0.5pt, 8=1pt, 12=1.5pt)
    color: hex color string
    """
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = f'w:{edge}'
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key, attr in [('val', 'w:val'), ('color', 'w:color'), ('sz', 'w:sz'), ('space', 'w:space')]:
                if key in edge_data:
                    element.set(qn(attr), str(edge_data[key]))

def set_table_booktabs(table, col_widths, align_cols=None):
    """Apply IEEE booktabs styling (top rule, mid rule, bottom rule, no vertical rules)."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    num_rows = len(table.rows)
    num_cols = len(col_widths)
    
    # Repeat header row on every page
    trPr = table.rows[0]._tr.get_or_add_trPr()
    trPr.append(OxmlElement('w:tblHeader'))

    for r_idx, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement('w:cantSplit'))
        for c_idx, cell in enumerate(row.cells):
            cell.width = col_widths[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=30, bottom=30, left=50, right=50)
            
            # borders
            border_kwargs = {
                'left': {'val': 'none'},
                'right': {'val': 'none'},
            }
            if r_idx == 0:
                border_kwargs['top'] = {'val': 'single', 'sz': '12', 'color': '000000'}
                border_kwargs['bottom'] = {'val': 'single', 'sz': '8', 'color': '000000'}
                set_cell_shading(cell, "F0F4F8")
            elif r_idx == num_rows - 1:
                border_kwargs['bottom'] = {'val': 'single', 'sz': '12', 'color': '000000'}
                border_kwargs['top'] = {'val': 'none'}
            else:
                border_kwargs['top'] = {'val': 'none'}
                border_kwargs['bottom'] = {'val': 'none'}
            
            set_cell_border(cell, **border_kwargs)

def add_columns_to_section(section, num_cols=2, space_pt=18):
    sectPr = section._sectPr
    cols = sectPr.find(qn('w:cols'))
    if cols is None:
        cols = OxmlElement('w:cols')
        sectPr.append(cols)
    cols.set(qn('w:num'), str(num_cols))
    cols.set(qn('w:space'), str(int(space_pt * 20)))

def format_run(run, font_name="Times New Roman", size_pt=10, bold=False, italic=False, color_rgb=(0,0,0)):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color_rgb)

def add_p(doc, text="", font_name="Times New Roman", size_pt=9.5, bold=False, italic=False,
          align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=1.5, line_spacing=1.0):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if text:
        r = p.add_run(text)
        format_run(r, font_name, size_pt, bold, italic)
    return p

def add_heading_1(doc, title_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(title_text)
    format_run(r, "Times New Roman", 10, bold=True)
    return p

def add_heading_2(doc, title_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(title_text)
    format_run(r, "Times New Roman", 9.5, bold=True, italic=True)
    return p

def add_equation(doc, eq_text, eq_num):
    """Centered equation with flush-right equation number."""
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # 2 columns in a single column measure (~3.45 in) or full width (~7.1 in)
    # Detect current section columns
    col_w1 = Inches(2.95)
    col_w2 = Inches(0.50)
    table.rows[0].cells[0].width = col_w1
    table.rows[0].cells[1].width = col_w2
    
    p_eq = table.rows[0].cells[0].paragraphs[0]
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq.paragraph_format.space_before = Pt(2)
    p_eq.paragraph_format.space_after = Pt(2)
    r_eq = p_eq.add_run(eq_text)
    format_run(r_eq, "Cambria Math", 9.5, italic=True)
    
    p_num = table.rows[0].cells[1].paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_num.paragraph_format.space_before = Pt(2)
    p_num.paragraph_format.space_after = Pt(2)
    r_num = p_num.add_run(f"({eq_num})")
    format_run(r_num, "Times New Roman", 9.5, bold=False)
    
    for c in table.rows[0].cells:
        set_cell_border(c, top={'val':'none'}, bottom={'val':'none'}, left={'val':'none'}, right={'val':'none'})
        set_cell_margins(c, top=20, bottom=20, left=20, right=20)
    
    # spacer after
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(2)

def add_figure(doc, img_path, caption_title, caption_text, width_inches=3.4):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    if Path(img_path).exists():
        run_img = p_img.add_run()
        run_img.add_picture(str(img_path), width=Inches(width_inches))
    else:
        r_miss = p_img.add_run(f"[{Path(img_path).name} not found]")
        format_run(r_miss, "Times New Roman", 9, italic=True)
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(7)
    r_title = p_cap.add_run(caption_title + " ")
    format_run(r_title, "Times New Roman", 8.5, bold=True)
    r_text = p_cap.add_run(caption_text)
    format_run(r_text, "Times New Roman", 8.5)
    return p_cap

def build_paper_document():
    doc = docx.Document()
    
    # Section 1: Title, Authors, Abstract, Keywords (1 column)
    s1 = doc.sections[0]
    s1.page_width = Inches(8.27)
    s1.page_height = Inches(11.69)
    s1.top_margin = Inches(0.75)
    s1.bottom_margin = Inches(1.0)
    s1.left_margin = Inches(0.56)
    s1.right_margin = Inches(0.56)
    add_columns_to_section(s1, num_cols=1)
    
    # Title
    p_title = add_p(doc, "ViT-XAI-Lyo: An Attention-Explainable Vision Transformer Framework for Lyophilized Cake Defect Classification with Regulatory-Aligned Interpretability",
                    size_pt=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=10, line_spacing=1.1)
    
    # Authors
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_before = Pt(2)
    p_auth.paragraph_format.space_after = Pt(2)
    for name, affil in [("Mathan M", "1"), ("Chandru S", "1"), ("Murugayesu A", "1"), ("Dr. Shanmuga sundar G", "2")]:
        r = p_auth.add_run(name)
        format_run(r, "Times New Roman", 11, bold=True)
        r_sup = p_auth.add_run(affil)
        format_run(r_sup, "Times New Roman", 9, bold=True)
        r_sup.font.superscript = True
        if name != "Dr. Shanmuga sundar G":
            p_auth.add_run(",   ")
            
    # Affiliations
    p_aff1 = add_p(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=1, space_after=1)
    r_a1 = p_aff1.add_run("1 Department of CSE (AI&ML), Sri Sai Ram Institute of Technology, Chennai, India\n")
    format_run(r_a1, "Times New Roman", 9.5, italic=True)
    r_em = p_aff1.add_run("{sit24am016, sit24am025, sit24am039}@sairamtap.edu.in")
    format_run(r_em, "Courier New", 8.5)
    
    p_aff2 = add_p(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=1, space_after=8)
    r_a2 = p_aff2.add_run("2 Department of Mechanical Engineering, Sri Sai Ram Engineering College, Chennai, India")
    format_run(r_a2, "Times New Roman", 9.5, italic=True)

    # Abstract Box
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.space_before = Pt(4)
    p_abs.paragraph_format.space_after = Pt(3)
    p_abs.paragraph_format.left_indent = Inches(0.15)
    p_abs.paragraph_format.right_indent = Inches(0.15)
    
    r_abs_h = p_abs.add_run("Abstract— ")
    format_run(r_abs_h, "Times New Roman", 9, bold=True, italic=True)
    
    abs_text = (
        "Lyophilized (freeze-dried) biologic therapeutics — monoclonal antibodies, recombinant proteins and "
        "mRNA vaccines — constitute the most fragile sector of biopharmaceutical manufacturing. Structural defects "
        "in the cake matrix, notably gross collapse, localized meltback and surface skinning, compromise reconstitution "
        "kinetics, potency and sterility. Inspection remains predominantly manual and subjective, while deep-learning "
        "automated visual inspection (AVI) operates as an uninterpretable black box that cannot supply the auditable "
        "causal evidence required under 21 CFR Part 11 and the FDA's January 2025 draft guidance on artificial intelligence "
        "in drug and biological product decision-making.\n\n"
        "This paper presents ViT-XAI-Lyo, an attention-explainable Vision Transformer framework for lyophilized cake defect "
        "classification with built-in regulatory traceability. A ViT-B/16 backbone models non-local dependencies across the "
        "whole vial geometry; a dual explainability stream pairs Attention Rollout with token-adapted Gradient-weighted Class "
        "Activation Mapping (Grad-CAM); a structured audit-metadata schema and a defined batch-disposition risk score place "
        "every prediction in a human-in-the-loop quality-assurance workflow; and a regulatory crosswalk maps each validation "
        "artifact onto the seven steps of the FDA credibility assessment framework.\n\n"
        "Because proprietary vial imagery was unavailable, the framework was evaluated end-to-end on a public industrial "
        "proxy benchmark (MVTec AD Bottle, n = 292) mapped to a four-class lyophilization taxonomy. Against a fully fine-tuned "
        "ResNet-50 baseline under identical frozen stratified splits and three random seeds, ViT-B/16 reached a test macro-F1 "
        "of 0.843 ± 0.100 versus 0.792 ± 0.015, while updating only 16.7% of its parameters (14.3 M of 85.8 M) against the "
        "baseline's 100%. The difference is not statistically significant (paired t-test t = 0.79, p = 0.513, n = 3 seeds; "
        "image-level paired bootstrap p = 0.542, 95% CI [-0.167, +0.254]), and we report the two architectures as performing "
        "comparably. Quantitative faithfulness evaluation against pixel-level ground-truth masks and against a random-saliency "
        "control found token-adapted Grad-CAM the more reliable attribution method (44% pointing-game accuracy, 95% CI [19%, 73%]), "
        "while Attention Rollout provided complementary global coverage at low localization precision (11% accuracy, 95% CI [2%, 44%]). "
        "We therefore locate the contribution of ViT-XAI-Lyo in quantified, regulator-mappable interpretability at accuracy parity, "
        "rather than in a classification-accuracy advantage."
    )
    r_abs_t = p_abs.add_run(abs_text)
    format_run(r_abs_t, "Times New Roman", 9)
    
    # Keywords
    p_kw = doc.add_paragraph()
    p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_kw.paragraph_format.space_before = Pt(3)
    p_kw.paragraph_format.space_after = Pt(10)
    p_kw.paragraph_format.left_indent = Inches(0.15)
    p_kw.paragraph_format.right_indent = Inches(0.15)
    
    r_kw_h = p_kw.add_run("Keywords— ")
    format_run(r_kw_h, "Times New Roman", 9, bold=True, italic=True)
    r_kw_t = p_kw.add_run("Vision Transformers, Explainable AI, Lyophilization, Defect Classification, Attention Rollout, Grad-CAM, "
                          "Pharmaceutical Quality Control, FDA AI Credibility Framework, Audit Traceability, Biologic Drug Manufacturing.")
    format_run(r_kw_t, "Times New Roman", 9, italic=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Section 2: Body text (2 columns)
    # ─────────────────────────────────────────────────────────────────────────
    s2 = doc.add_section(WD_SECTION.CONTINUOUS)
    s2.top_margin = Inches(0.75)
    s2.bottom_margin = Inches(1.0)
    s2.left_margin = Inches(0.56)
    s2.right_margin = Inches(0.56)
    add_columns_to_section(s2, num_cols=2, space_pt=18)
    
    # I. INTRODUCTION
    add_heading_1(doc, "I. INTRODUCTION")
    add_p(doc, "Lyophilization (freeze-drying) is the gold-standard stabilization route for delicate biopharmaceutical products, "
               "preserving temperature-sensitive antibodies, enzymes and vaccines by removing solvent through low-temperature sublimation "
               "under vacuum [1], [2]. The physical integrity of the resulting porous cake is a critical quality attribute (CQA): deviations "
               "in primary or secondary drying trigger structural failures such as cake collapse (loss of the structural matrix where the "
               "product temperature exceeds the collapse temperature, closely related to the glass transition temperature of the maximally "
               "freeze-concentrated solute Tg') [3], meltback (localized interstitial melting yielding dense vitreous cores), and skinning "
               "(an impermeable surface crust hindering vapour escape) [4]. These defects impede reconstitution, promote protein aggregation "
               "and risk patient safety.")
    
    add_p(doc, "Despite stringent regulatory standards, inspection of lyophilized vials on commercial fill-finish lines remains heavily "
               "reliant on manual visual inspection (MVI), which suffers from operator variability, visual fatigue, limited throughput and "
               "qualitative subjectivity [5]. Automated visual inspection (AVI) using deep learning has been introduced industrially — including "
               "convolutional detectors patented specifically for lyophilized drug products [6], [7], a YOLOv11-derived detector for lyophilized "
               "vial defects [8], and a YOLOv8-derived detector for foreign objects in lyophilized powder [9] — and offline tomographic assessment "
               "of freeze-dried structure has been demonstrated [10]. Adoption inside regulated manufacturing is nonetheless curtailed by the "
               "black-box dilemma: quality assurance (QA) personnel and regulators cannot authorize automated batch disposition without auditable "
               "visual evidence tracing why a vial was flagged [11].")
    
    add_p(doc, "Vision Transformers (ViTs) [12] offer a relevant architectural shift. Unlike convolutional filters operating over restricted local "
               "receptive fields, self-attention relates all image patches simultaneously [13]. This suits lyophilized cakes, where collapse and "
               "meltback manifest as diffuse, distributed changes across the matrix rather than isolated edge anomalies. Crucially, attention weights "
               "constitute an internal, inspectable record of feature relevance [14], and recent work continues to refine their interpretability [15].")
    
    add_p(doc, "The gap this paper addresses is therefore not the application of deep learning to vial inspection — prior art [6]–[9] reached that "
               "domain first — but the absence of any work that quantifies whether the explanations such a system produces are faithful, and that "
               "maps that evidence onto a regulatory credibility framework. Qualitative heatmaps are routinely displayed in the industrial-inspection "
               "literature; their faithfulness is rarely measured [16], and never, to our knowledge, in a pharmaceutical context with an explicit "
               "alignment to the FDA's AI credibility expectations [17].")
    
    add_p(doc, "The key contributions of this work are:")
    
    contribs = [
        ("1. Pharma-specific defect taxonomy: ", "We formalize a four-class lyophilized cake scheme mapped to freeze-drying degradation mechanisms: Defect-Free, Collapse, Meltback, Skinning (§IV.A)."),
        ("2. Global-attention classification architecture: ", "We deploy ViT-B/16 to capture long-range spatial correlations across the porous cake matrix, and show that it attains parity with a fully fine-tuned ResNet-50 while updating only 16.7% of its parameters (§III.B, §V.A)."),
        ("3. Dual quantitative explainability with a random control: ", "We integrate Attention Rollout [14] and token-adapted Grad-CAM [18], and evaluate both using Deletion AUC, Insertion AUC, Pointing-Game accuracy against pixel-level masks, and cross-method spatial agreement — benchmarked against a random-saliency control that is absent from prior industrial-XAI studies (§III.D–E, §V.E)."),
        ("4. Audit traceability and batch disposition: ", "We specify a structured audit-metadata schema and a batch-disposition risk score that combines classification confidence, attribution faithfulness and defect severity into an assistive recommendation under QA sign-off (§III.F–H)."),
        ("5. Honest empirical benchmarking: ", "We benchmark against ResNet-50 under identical frozen splits with multi-seed evaluation, and report confidence intervals and two independent significance tests, both indicating the architectures are not separable at this sample size (§V.B)."),
        ("6. Regulatory crosswalk: ", "We map every validation artifact onto the seven steps of the FDA January 2025 credibility assessment framework [17] (§VI)."),
    ]
    for c_h, c_t in contribs:
        p_c = doc.add_paragraph()
        p_c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_c.paragraph_format.left_indent = Inches(0.12)
        p_c.paragraph_format.space_before = Pt(1)
        p_c.paragraph_format.space_after = Pt(2)
        r_h = p_c.add_run(c_h)
        format_run(r_h, "Times New Roman", 9.5, bold=True)
        r_t = p_c.add_run(c_t)
        format_run(r_t, "Times New Roman", 9.5)

    # II. RELATED WORK
    add_heading_1(doc, "II. RELATED WORK AND PRIOR-ART MAPPING")
    add_heading_2(doc, "A. Deep Learning for Lyophilized Product Inspection")
    add_p(doc, "The closest prior art is domain-specific. Genentech holds patents covering convolutional defect detection in lyophilized drug products "
               "[6], [7], establishing both industrial interest and the CNN-centric status quo. Jiang et al. [8] proposed SAF-YOLO, a YOLOv11 derivative "
               "with spectral and attention fusion for defect detection in lyophilized vials, reporting strong detection performance on small, weak-signal "
               "defects. Xu et al. [9] proposed CSLNet, a YOLOv8 derivative targeting surface foreign objects in lyophilized powder. Xu et al. [19] "
               "describe a broader vision-based inspection system for pharmaceutical production lines. Offline tomographic assessment of structural "
               "defects in freeze-dried pharmaceuticals [10] achieves high structural fidelity but requires costly, non-inline instrumentation incapable "
               "of real-time screening at commercial line speeds of 400–600 vials/minute. None of these prior systems quantifies attribution faithfulness, "
               "and none maps evidence to regulatory guidelines.")
    
    add_heading_2(doc, "B. Vision Transformers in Industrial Quality Control")
    add_p(doc, "Pure self-attention architectures match or exceed CNN performance in visual recognition [12], [13]. Hybrid CNN–Transformer models have "
               "been applied to strip-steel surface classification [20] and, with explainability, to radiographic weld inspection [21], the latter being "
               "notable for coupling a hybrid CNN-ViT with misclassification-aware explanation. Transformer approaches to semiconductor wafer-map defects "
               "have also been explored [22]. These works establish that global self-attention helps on diffuse, irregular textures, but they do not "
               "address pharmaceutical quality control and do not evaluate whether their attention maps satisfy regulatory verification expectations.")

    add_heading_2(doc, "C. Explainable AI and Faithfulness Evaluation")
    add_p(doc, "Vision XAI is dominated by post-hoc gradient methods such as Grad-CAM [18] and architecture-native techniques such as Attention Rollout [14]; "
               "GMAR [15] extends rollout with gradient-driven head weighting. Perturbation-based faithfulness metrics derive from RISE [16]. The recurring "
               "methodological weakness in applied work is the absence of a random baseline: deletion-curve metrics are sensitive to the spatial dispersion "
               "of a saliency map independently of its correctness, because scattering perturbations across an image degrades it faster than removing "
               "one compact region. We address this directly in §V.E.")

    add_heading_2(doc, "D. Positioning")
    add_p(doc, "Table I maps the prior art against ViT-XAI-Lyo. Our novelty is not first application of deep learning, or even of attention, to vial inspection. "
               "It is the first framework in this domain to quantify explanation faithfulness against ground truth and a random control, and to operationalize "
               "that evidence as a regulatory credibility artifact with an auditable disposition workflow.")

    # Table I: Full-width section starting at top of page for clean float placement
    s_t1 = doc.add_section(WD_SECTION.NEW_PAGE)
    add_columns_to_section(s_t1, num_cols=1)
    
    p_t1_h = add_p(doc, "TABLE I. PRIOR-ART COMPARISON ACROSS LYOPHILIZATION AND INDUSTRIAL XAI",
                   size_pt=8.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=1)
    p_t1_sub = add_p(doc, "('Quant. XAI' denotes numerically evaluated attribution faithfulness against ground truth / controls, not qualitative heatmaps.)",
                     size_pt=8, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=4)
    
    t1_headers = ["Reference", "Architecture", "Task / Domain", "XAI Method", "Quant. XAI", "Reg. Mapping", "Key Limitation vs. Ours"]
    t1_rows = [
        ["[6], [7] Genentech", "CNN", "Lyophilized drug products", "None", "No", "Industrial", "Proprietary, unreproducible; no interpretability"],
        ["[8] SAF-YOLO (2025)", "YOLOv11 + attn", "Lyophilized vial defects", "None", "No", "None", "Detection only; no attribution evidence"],
        ["[9] CSLNet (2025)", "YOLOv8 derivative", "Foreign objects, lyo powder", "None", "No", "None", "Foreign-object focus, not cake morphology"],
        ["[10] Müller et al. (2024)", "Robotic µ-CT + ML", "Freeze-dried structural defects", "None", "No", "None", "Offline; non-inline; high instrumentation cost"],
        ["[21] Parmar et al. (2026)", "Hybrid CNN-ViT", "Radiographic weld inspection", "Misclassification XAI", "No", "None", "Welding domain; faithfulness not quantified"],
        ["[14], [15] Abnar / Jo", "ViT", "General image recognition", "Rollout / GMAR", "Partial", "None", "Method papers; no industrial or cGMP evaluation"],
        ["Ours (ViT-XAI-Lyo)", "ViT-B/16 vs ResNet-50", "Lyo cake defect classification", "Rollout + Grad-CAM", "Yes (incl. random control)", "FDA 7-step crosswalk", "Evaluated on proxy benchmark; 16.7% param tuning"],
    ]
    
    t1 = doc.add_table(rows=len(t1_rows)+1, cols=len(t1_headers))
    for c_idx, h in enumerate(t1_headers):
        p_c = t1.rows[0].cells[c_idx].paragraphs[0]
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p_c.add_run(h)
        format_run(r, "Times New Roman", 8, bold=True)
    for r_idx, row_data in enumerate(t1_rows):
        is_ours = (r_idx == len(t1_rows) - 1)
        for c_idx, val in enumerate(row_data):
            p_c = t1.rows[r_idx+1].cells[c_idx].paragraphs[0]
            p_c.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx in [2, 3, 6] else WD_ALIGN_PARAGRAPH.CENTER
            r = p_c.add_run(val)
            format_run(r, "Times New Roman", 7.5, bold=is_ours)
            if is_ours:
                set_cell_shading(t1.rows[r_idx+1].cells[c_idx], "EBF3FA")
                
    t1_widths = [Inches(1.0), Inches(1.05), Inches(1.15), Inches(0.95), Inches(0.65), Inches(0.85), Inches(1.50)]
    set_table_booktabs(t1, t1_widths)
    
    # Switch back to 2 columns
    s_b2 = doc.add_section(WD_SECTION.CONTINUOUS)
    add_columns_to_section(s_b2, num_cols=2, space_pt=18)
    add_columns_to_section(s_b2, num_cols=2, space_pt=18)
    
    # III. SYSTEM ARCHITECTURE
    add_heading_1(doc, "III. SYSTEM ARCHITECTURE AND METHODOLOGY")
    add_p(doc, "The ViT-XAI-Lyo pipeline comprises six modules: (i) non-invasive vial image acquisition and pre-processing, (ii) Vision Transformer "
               "feature extraction, (iii) multi-class defect classification with class-imbalance weighting, (iv) dual attention explainability, "
               "(v) audit-metadata capture and batch-disposition risk scoring, and (vi) an interpretable reporting module feeding human-in-the-loop "
               "QA disposition. Modules (i)–(iv) are evaluated empirically in §V; modules (v)–(vi) are specified here and instantiated in the "
               "released demo interface.")
    
    add_heading_2(doc, "A. Operating Context")
    add_p(doc, "ViT-XAI-Lyo is specified as an assistive pre-QA screening aid, not an autonomous disposition authority. Every prediction is "
               "accompanied by its attribution evidence and is subject to QA sign-off. The system consumes only non-invasive vial imagery and "
               "structured inspection metadata; it requires no proprietary formulation data and no patient records. This context of use fixes the "
               "model-risk tier used in §VI.")
    
    add_heading_2(doc, "B. Vision Transformer Architecture")
    add_p(doc, "Let an input vial image be X ∈ R^{H × W × C} with H = W = 224, C = 3. The image is partitioned into non-overlapping patches "
               "x_p ∈ R^{N × (P² C)} with patch size P = 16, yielding:")
    add_equation(doc, "N = (H · W) / P² = (224 × 224) / (16 × 16) = 196", "1")
    
    add_p(doc, "Each flattened patch is projected to a D-dimensional latent vector (D = 768) by a learnable linear map E ∈ R^{(P² C) × D}. A learnable "
               "classification token x_class ∈ R^{1 × D} is prepended and learnable 1-D position embeddings E_pos ∈ R^{(N+1) × D} are added:")
    add_equation(doc, "z_0 = [x_class; x_p^1 E; ...; x_p^N E] + E_pos", "2")
    
    add_p(doc, "The sequence passes through L = 12 encoder blocks, each applying Multi-Head Self-Attention (MSA) and an MLP with pre-Layer-Normalization "
               "(LN) and residual connections:")
    add_equation(doc, "z'_l = MSA(LN(z_{l-1})) + z_{l-1},    l = 1, ..., L", "3")
    add_equation(doc, "z_l = MLP(LN(z'_l)) + z'_l,        l = 1, ..., L", "4")
    
    add_p(doc, "Within each of the h = 12 heads, queries, keys and values are formed with projections W_Q, W_K, W_V ∈ R^{D × d_k}, d_k = D/h = 64:")
    add_equation(doc, "A_{l,i} = softmax( (Q K^T) / sqrt(d_k) ),    head_{l,i} = A_{l,i} V", "5")
    
    add_p(doc, "The classification token at the final layer is normalized and mapped to K = 4 classes:")
    add_equation(doc, "y_hat = softmax( W_head · LN(z_L^0) )", "6")

    add_heading_2(doc, "C. Class-Imbalance Weighted Loss")
    add_p(doc, "Class imbalance is intrinsic to pharmaceutical manufacturing: most vials are defect-free while critical defects are rare process deviations. "
               "We employ an inverse-frequency weighted cross-entropy loss:")
    add_equation(doc, "L = - sum_{k=1}^K w_k · y_k · log(y_hat_k),    w_k = ( (1/N_k) / sum_{j=1}^K (1/N_j) ) · K", "7")
    add_p(doc, "where N_k is the number of training samples of class k, computed exclusively on the training partition.")

    add_heading_2(doc, "D. Explainability Module 1: Attention Rollout")
    add_p(doc, "Raw final-layer attention does not reflect token mixing through earlier layers. We compute Attention Rollout [14] by averaging heads per layer:")
    add_equation(doc, "A_bar_l = (1 / h) sum_{i=1}^h A_{l,i}", "8")
    add_p(doc, "augmenting with the identity matrix to account for residual connections and row-normalizing:")
    add_equation(doc, "A_hat_l = 0.5 · A_bar_l + 0.5 · I,    sum_j A_hat_{l,ij} = 1", "9")
    add_p(doc, "and taking the recursive product through all L layers:")
    add_equation(doc, "R = A_hat_L · A_hat_{L-1} ··· A_hat_1", "10")
    add_p(doc, "The CLS row, excluding the CLS token index, gives the spatial patch relevance map:")
    add_equation(doc, "M_rollout = R[0, 1:] ∈ R^{196} -> 14 × 14 -> 224 × 224", "11")

    add_heading_2(doc, "E. Explainability Module 2: ViT-Adapted Grad-CAM")
    add_p(doc, "For an independent gradient-based check, we adapt Grad-CAM [18] to the transformer. Features are taken at the pre-attention normalization "
               "of the final block, T ∈ R^{B × (N+1) × D}; the CLS token is discarded and the remaining N = 196 tokens are reshaped to F ∈ R^{B × D × 14 × 14}. "
               "Target-class gradients are spatially pooled:")
    add_equation(doc, "alpha_k^c = (1 / 196) sum_{i=1}^{14} sum_{j=1}^{14} ( ∂y^c / ∂F_{ij}^k )", "12")
    add_p(doc, "and combined with a ReLU rectification:")
    add_equation(doc, "M_Grad-CAM = ReLU( sum_k alpha_k^c · F^k )", "13")

    add_heading_2(doc, "F. Audit Metadata Schema and Traceability")
    add_p(doc, "Each inspection record carries a structured metadata envelope supporting 21 CFR Part 11-style traceability: vial identifier, batch "
               "identifier, capture geometry, severity label, SHA-256 image hash, model identifier, checkpoint hash, software dependencies, random seed, "
               "predicted class, posterior vector, attribution maps, faithfulness metrics, disposition recommendation, and the identity and decision of "
               "the signing QA reviewer. Records are append-only: a re-inspection creates a new record referencing its predecessor rather than mutating it.")

    add_heading_2(doc, "G. Batch-Disposition Risk Score")
    add_p(doc, "To convert a per-vial prediction into an assistive recommendation, we define a disposition risk score refusing reliance on confidence alone. "
               "For vial v with predicted class c, posterior y_hat_c, severity weight σ_c ∈ [0,1] and attribution faithfulness φ_v:")
    add_equation(doc, "rho_v = sigma_c · [ lambda_1 · y_hat_c + lambda_2 · phi_v + lambda_3 · (1 - H(y_hat)/log K) ]", "14")
    add_p(doc, "where H is Shannon entropy rewarding decisive predictions, and φ_v is the per-image insertion-minus-deletion margin of the Grad-CAM map "
               "normalized to [0,1]. Severity weights follow freeze-drying failure physics: σ = 0 for defect-free, 0.6 meltback, 0.8 skinning, 1.0 collapse, "
               "with λ = (0.5, 0.3, 0.2). A confident prediction with unfaithful explanation receives a lower score, triggering mandatory human escalation.")

    add_heading_2(doc, "H. Interpretable Reporting Module")
    add_p(doc, "The reporting module presents for each vial: predicted class, posterior distribution, original image, both attribution overlays "
               "side by side, numerical faithfulness metrics, and the disposition score recommendation. Disagreement between Rollout and Grad-CAM is "
               "explicitly surfaced as diagnostic evidence for QA personnel.")

    # Insert Fig. 1
    add_figure(doc, FIG_DIR / "fig1_system_architecture.png", "Fig. 1. ViT-XAI-Lyo system architecture.",
               "Six-stage end-to-end pipeline: non-invasive vial acquisition and 16×16 patch partition (N = 196, D = 768); ViT-B/16 encoder (L = 12 blocks, h = 12 heads); "
               "class-weighted softmax classifier; dual explainability stream producing Attention Rollout and token-adapted Grad-CAM maps; "
               "audit-metadata capture with batch-disposition risk scoring; and the interpretable reporting module feeding human-in-the-loop QA sign-off.",
               width_inches=3.4)

    # IV. EXPERIMENTAL SETUP
    add_heading_1(doc, "IV. EXPERIMENTAL SETUP AND EVALUATION PROTOCOL")
    add_heading_2(doc, "A. Dataset and Proxy-Domain Rationale")
    add_p(doc, "No real lyophilized vial imagery was available to this study due to cGMP confidentiality and lack of pilot-scale lyophilizer access. "
               "Rather than report on synthetic images, which would not constitute verifiable evaluation, the complete protocol was executed on "
               "the MVTec AD Bottle benchmark [23]. It shares salient optical geometry — transparent cylindrical glass with specular boundary reflections "
               "under controlled lighting — while being fully public and reproducible. It is an optical proxy only: bottle defects are not lyophilization "
               "defects, and Table II establishes a structural analogy.")

    # Table II
    p_t2_h = add_p(doc, "TABLE II. DATASET COMPOSITION, CLASS MAPPING AND FROZEN PARTITION",
                   size_pt=8.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=1)
    t2_headers = ["MVTec AD Bottle", "Mapped Class", "Analogy", "Total", "Train", "Val", "Test"]
    t2_rows = [
        ["good", "Defect-Free", "Intact cake matrix", "229", "160", "34", "35"],
        ["broken_large", "Collapse", "Gross matrix loss", "20", "14", "3", "3"],
        ["broken_small", "Meltback", "Local discontinuity", "22", "15", "4", "3"],
        ["contamination", "Skinning", "Surface-layer crust", "21", "15", "3", "3"],
        ["Total", "—", "—", "292", "204", "44", "44"],
    ]
    t2 = doc.add_table(rows=len(t2_rows)+1, cols=len(t2_headers))
    for c_idx, h in enumerate(t2_headers):
        p_c = t2.rows[0].cells[c_idx].paragraphs[0]
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p_c.add_run(h)
        format_run(r, "Times New Roman", 7.5, bold=True)
    for r_idx, row_data in enumerate(t2_rows):
        is_tot = (r_idx == len(t2_rows) - 1)
        for c_idx, val in enumerate(row_data):
            p_c = t2.rows[r_idx+1].cells[c_idx].paragraphs[0]
            p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p_c.add_run(val)
            format_run(r, "Times New Roman", 7.5, bold=is_tot)
            if is_tot:
                set_cell_shading(t2.rows[r_idx+1].cells[c_idx], "F0F4F8")
    t2_widths = [Inches(0.70), Inches(0.60), Inches(0.75), Inches(0.35), Inches(0.35), Inches(0.35), Inches(0.35)]
    set_table_booktabs(t2, t2_widths)
    add_p(doc, "", space_before=0, space_after=2)

    add_heading_2(doc, "B. Partitioning Protocol")
    add_p(doc, "Partitions were assigned by class-stratified random sampling (70/15/15), frozen with split seed 42, and reused across all models "
               "and seeds. The proxy benchmark contains single views without batch metadata; true multi-view vial grouping is deferred to validation "
               "on industrial lines.")

    add_heading_2(doc, "C. Training Configuration")
    add_p(doc, "Both architectures were trained with the objective in (7), 224×224 resolution, AdamW optimizer (β1 = 0.9, β2 = 0.999, weight decay 10^-4), "
               "30 epochs, gradient clipping at 1.0, and ImageNet pre-trained weights on an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) under "
               "PyTorch 2.5.1 + CUDA 12.1. The configurations are intentionally asymmetric to reflect realistic deployment constraints:\n"
               "• ResNet-50 (baseline): All 23.52 M parameters fine-tuned (100%); learning rate η = 3×10^-4; cosine annealing; batch size 8; AMP enabled.\n"
               "• ViT-B/16 (proposed): Patch embed and encoder blocks 1–10 frozen; only blocks 11–12, final LN and head fine-tuned (14.33 M of 85.80 M parameters, "
               "16.7%); η = 1×10^-4; warmup + cosine decay; batch size 4; fp32 precision.\n"
               "• Augmentation (both): Horizontal flip (p = 0.5), vertical flip (p = 0.2), colour jitter (brightness 0.15, contrast 0.15, saturation 0.10).\n"
               "• Seeds: Evaluated over seeds {42, 123, 7}.")

    add_heading_2(doc, "D. Faithfulness Evaluation Metrics")
    add_p(doc, "1. Deletion AUC: Progressive removal of top salient pixels replacing with image mean over 20 steps (lower is better).\n"
               "2. Insertion AUC: Progressive restoration of top salient pixels from a mean baseline (higher is better).\n"
               "3. Random-saliency control: Uniform random heatmap passed through identical perturbation sweeps to test for diffuseness bias.\n"
               "4. Pointing-Game accuracy: Hit rate where peak saliency lands inside pixel-level ground-truth masks (9 masked defect images).\n"
               "5. Cross-method spatial agreement: IoU between top-50th-percentile supports of Rollout and Grad-CAM.")

    # V. RESULTS
    add_heading_1(doc, "V. RESULTS AND DISCUSSION")
    add_heading_2(doc, "A. Classification Benchmark")
    add_p(doc, "Table III summarizes classification performance on the held-out test set (n = 44).")

    # Table III
    p_t3_h = add_p(doc, "TABLE III. CLASSIFICATION PERFORMANCE ON TEST SET (n = 44, MEAN ± S.D. OVER 3 SEEDS)",
                   size_pt=8.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=1)
    t3_headers = ["Architecture", "Trainable Params", "Macro-F1", "Macro-Prec.", "Macro-Rec.", "Accuracy"]
    t3_rows = [
        ["ResNet-50", "23.52 M (100%)", "0.792 ± 0.015", "0.883 ± 0.042", "0.778 ± 0.039", "0.939"],
        ["ViT-B/16 (ours)", "14.33 M (16.7%)", "0.843 ± 0.100", "0.898 ± 0.055", "0.833 ± 0.118", "0.955"],
    ]
    t3 = doc.add_table(rows=3, cols=6)
    for c_idx, h in enumerate(t3_headers):
        p_c = t3.rows[0].cells[c_idx].paragraphs[0]
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p_c.add_run(h)
        format_run(r, "Times New Roman", 7.5, bold=True)
    for r_idx, row_data in enumerate(t3_rows):
        for c_idx, val in enumerate(row_data):
            p_c = t3.rows[r_idx+1].cells[c_idx].paragraphs[0]
            p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p_c.add_run(val)
            format_run(r, "Times New Roman", 7.5, bold=(r_idx==1))
            if r_idx == 1:
                set_cell_shading(t3.rows[r_idx+1].cells[c_idx], "EBF3FA")
    t3_widths = [Inches(0.75), Inches(0.65), Inches(0.55), Inches(0.50), Inches(0.50), Inches(0.45)]
    set_table_booktabs(t3, t3_widths)
    add_p(doc, "", space_before=0, space_after=2)

    add_p(doc, "Per-seed macro-F1 was 0.812 / 0.782 / 0.780 for ResNet-50 and 0.914 / 0.914 / 0.701 for ViT-B/16 (seeds 42 / 123 / 7). "
               "Fig. 2 plots the training and validation trajectories. Both models drive training loss to approximately zero within 30 epochs. "
               "Validation loss achieves its minimum early (epochs 2–7 for ResNet-50, 1–2 for ViT-B/16) and increases thereafter, reflecting "
               "the small sample regime. Checkpoint selection on validation macro-F1 selects early checkpoints (epochs 3–7 for ViT-B/16, 7–13 for ResNet-50).")

    # Fig. 2
    add_figure(doc, FIG_DIR / "fig2_training_curves.png", "Fig. 2. Training and validation dynamics over 30 epochs.",
               "(a) Cross-entropy loss; (b) macro-F1. Solid lines training, dashed validation; blue is ResNet-50, red is ViT-B/16 across seeds 42, 123, 7. "
               "Both memorize the training set; validation macro-F1 plateaus stably for ViT-B/16.",
               width_inches=3.4)

    add_heading_2(doc, "B. Statistical Analysis")
    add_p(doc, "We report two independent statistical tests:\n"
               "• Seed-level paired t-test (n = 3): t = 0.788, p = 0.513, mean diff +0.052 macro-F1, 95% CI [-0.231, +0.334].\n"
               "• Image-level paired bootstrap (10,000 resamples): p = 0.542, mean diff +0.058, 95% CI [-0.167, +0.254].\n"
               "ViT-B/16 is not statistically separable from ResNet-50 on this benchmark. Confidence intervals span zero and are multiple times "
               "wider than the observed effect. Per-seed differences were +0.102, +0.132, and -0.078. With df = 2, critical two-tailed t is 4.303, "
               "requiring ~0.28 macro-F1 difference for significance. We report the architectures as comparable.")

    add_heading_2(doc, "C. Per-Class Performance")
    add_p(doc, "Table IV presents per-class F1 with 2,000-resample bootstrap 95% confidence intervals.")

    # Table IV
    p_t4_h = add_p(doc, "TABLE IV. PER-CLASS PERFORMANCE (TEST SUPPORT: 35 DEFECT-FREE, 3 PER DEFECT CLASS)",
                   size_pt=8.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=1)
    t4_headers = ["Class", "Support", "ViT-B/16 F1 [95% CI]", "ViT Rec.", "ResNet-50 F1 [95% CI]", "ResNet Rec."]
    t4_rows = [
        ["Defect-Free", "35", "0.991 [0.977, 1.000]", "1.000", "0.986 [0.968, 1.000]", "1.000"],
        ["Collapse", "3", "0.756 [0.000, 1.000]", "0.667", "0.758 [0.000, 0.933]", "0.889"],
        ["Meltback", "3", "0.794 [0.000, 1.000]", "0.889", "0.656 [0.000, 0.774]", "0.556"],
        ["Skinning", "3", "0.833 [0.000, 1.000]", "0.778", "0.767 [0.000, 1.000]", "0.667"],
    ]
    t4 = doc.add_table(rows=5, cols=6)
    for c_idx, h in enumerate(t4_headers):
        p_c = t4.rows[0].cells[c_idx].paragraphs[0]
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p_c.add_run(h)
        format_run(r, "Times New Roman", 7.5, bold=True)
    for r_idx, row_data in enumerate(t4_rows):
        for c_idx, val in enumerate(row_data):
            p_c = t4.rows[r_idx+1].cells[c_idx].paragraphs[0]
            p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p_c.add_run(val)
            format_run(r, "Times New Roman", 7.5)
    t4_widths = [Inches(0.65), Inches(0.40), Inches(0.75), Inches(0.40), Inches(0.75), Inches(0.45)]
    set_table_booktabs(t4, t4_widths)
    add_p(doc, "", space_before=0, space_after=2)

    add_p(doc, "Both models identify defect-free vials essentially perfectly. Because three test samples exist per defect class, a single misclassification "
               "shifts recall by 33.3%, yielding wide intervals that preclude definitive superiority claims.")

    add_heading_2(doc, "D. Error-Mode Analysis")
    add_p(doc, "Fig. 3 displays the row-normalized confusion matrices for the best checkpoint per architecture. The dominant error for both models is "
               "confusion within the collapse/meltback pair: ResNet-50 misassigns two meltback vials to collapse, while ViT-B/16 misassigns one collapse "
               "vial to meltback. Both represent matrix degradation differing primarily in spatial extent.")

    # Fig. 3
    add_figure(doc, FIG_DIR / "fig3_confusion_matrices.png", "Fig. 3. Confusion matrices for best checkpoints.",
               "Row-normalized confusion matrices over 44 test images (35 defect-free, 3 per defect class). Both models classify defect-free vials with 100% precision; "
               "errors concentrate on the collapse/meltback morphological boundary.",
               width_inches=3.4)

    add_heading_2(doc, "E. Quantitative Explainability Faithfulness")
    add_p(doc, "Table V presents attribution faithfulness against pixel-level masks and the random control.")

    # Table V
    p_t5_h = add_p(doc, "TABLE V. ATTRIBUTION FAITHFULNESS EVALUATION AND RANDOM CONTROL BENCHMARK",
                   size_pt=8.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=1)
    t5_headers = ["Method", "Deletion AUC ↓", "Insertion AUC ↑", "Ins. − Del.", "Pointing Game", "Wilson 95% CI", "p vs. Random"]
    t5_rows = [
        ["Attention Rollout", "0.255 ± 0.000", "0.237 ± 0.000", "-0.018", "1/9 (11%)", "[2%, 44%]", "1.000"],
        ["Grad-CAM (ViT)", "0.610 ± 0.000", "0.754 ± 0.000", "+0.144", "4/9 (44%)", "[19%, 73%]", "0.082"],
        ["Random Saliency (Control)", "0.632 ± 0.000", "0.628 ± 0.000", "-0.004", "0/9 (0%)", "[0%, 30%]", "—"],
    ]
    t5 = doc.add_table(rows=4, cols=7)
    for c_idx, h in enumerate(t5_headers):
        p_c = t5.rows[0].cells[c_idx].paragraphs[0]
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p_c.add_run(h)
        format_run(r, "Times New Roman", 7.5, bold=True)
    for r_idx, row_data in enumerate(t5_rows):
        for c_idx, val in enumerate(row_data):
            p_c = t5.rows[r_idx+1].cells[c_idx].paragraphs[0]
            p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p_c.add_run(val)
            format_run(r, "Times New Roman", 7.5, bold=(r_idx==1))
            if r_idx == 1:
                set_cell_shading(t5.rows[r_idx+1].cells[c_idx], "EBF3FA")
    t5_widths = [Inches(0.75), Inches(0.50), Inches(0.50), Inches(0.40), Inches(0.45), Inches(0.45), Inches(0.35)]
    set_table_booktabs(t5, t5_widths)
    add_p(doc, "", space_before=0, space_after=2)

    add_p(doc, "Cross-method spatial agreement (IoU of top-50% supports): 0.466 ± 0.096. Grad-CAM vs. Rollout pointing-game difference: Fisher exact p = 0.294.\n\n"
               "The random control clarifies the interpretation: uniform random saliency yields Deletion AUC 0.632 and Insertion AUC 0.628, an insertion-minus-deletion "
               "margin of -0.004, with 0/9 hits. Grad-CAM attains a margin of +0.144 and localizes 4 of 9 masked defects (44%, Wilson 95% CI [19%, 73%]; Fisher p = 0.082 vs control). "
               "Attention Rollout achieves a margin of -0.018 and localizes only 1 of 9 defects (11%, 95% CI [2%, 44%]; Fisher p = 1.000 vs control). "
               "While Rollout achieves a low Deletion AUC (0.255), the control indicates this is an artifact of diffuse support: scattered perturbations degrade global features "
               "faster than localized removal. Rollout indicates where the model looks broadly, while Grad-CAM localizes defects.")

    add_heading_2(doc, "F. Qualitative Comparison")
    add_p(doc, "Fig. 4 shows representative overlays. In defect-free vials attention spreads broadly. In collapse and meltback, Grad-CAM concentrates on compact "
               "boundaries, whereas Rollout covers the broader cake matrix.")

    # Fig. 4
    add_figure(doc, FIG_DIR / "fig4_heatmap_panel.png", "Fig. 4. Attention Rollout versus token-adapted Grad-CAM across defect classes.",
               "Columns: input vial image, Attention Rollout overlay, Grad-CAM overlay. Rows: Defect-Free, Collapse, Meltback, Skinning. "
               "Grad-CAM concentrates on localized structural boundaries, while Rollout captures global matrix involvement.",
               width_inches=3.4)

    add_heading_2(doc, "G. Batch-Disposition Risk Score Results")
    add_p(doc, "Applying (14) to the test set, defect-free vials receive ρ_v = 0 (σ = 0), and 9 of 44 vials are flagged. Over the flagged population, ρ_v ranges "
               "0.508–0.846 (mean 0.653). No flagged vial fell below the escalation threshold (ρ_v < 0.50), reflecting high model confidence on this test set.")

    add_heading_2(doc, "H. Mechanistic Architecture Comparison")
    add_p(doc, "Reviewer #4 requested an explanation of why the proposed approach performs differently from the baseline:\n"
               "1. Parameter efficiency under asymmetric fine-tuning: ViT-B/16 matched full ResNet-50 fine-tuning while updating only 16.7% of parameters (14.3 M vs 23.5 M). "
               "ImageNet-pretrained self-attention features transfer to cake morphology with minimal tuning, drastically lowering cGMP re-validation overhead.\n"
               "2. Inductive bias: CNN local receptive fields suit sharp boundary anomalies; self-attention captures long-range dependencies across the cake matrix.\n"
               "3. Quantitative interpretability: At accuracy parity, ViT supplies an intrinsic, inspectable attention mechanism validated through dual XAI.")

    # VI. REGULATORY CROSSWALK
    # Make Table VI full-width section spanning both columns at top of page
    s_t6 = doc.add_section(WD_SECTION.NEW_PAGE)
    add_columns_to_section(s_t6, num_cols=1)
    
    add_heading_1(doc, "VI. REGULATORY CROSSWALK TO THE FDA CREDIBILITY ASSESSMENT FRAMEWORK")
    add_p(doc, "The FDA's January 2025 draft guidance Considerations for the Use of Artificial Intelligence to Support Regulatory Decision-Making for Drug and "
               "Biological Products [17] establishes a 7-step credibility framework based on model risk. Table VI maps ViT-XAI-Lyo evidence onto these steps.")
    
    p_t6_h = add_p(doc, "TABLE VI. FDA SEVEN-STEP CREDIBILITY ASSESSMENT CROSSWALK FOR ViT-XAI-LYO",
                   size_pt=8.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=2)
    t6_headers = ["#", "FDA Step", "ViT-XAI-Lyo Implementation Evidence", "Section Reference"]
    t6_rows = [
        ["1", "Define the Question of Interest", "Does a vial exhibit cake collapse, meltback, or skinning requiring QA disposition?", "§I, §III.A"],
        ["2", "Define the Context of Use (COU)", "Assistive pre-QA screening with mandatory human sign-off; non-invasive imagery; no autonomous batch release.", "§III.A, §III.G"],
        ["3", "Assess AI Model Risk", "Medium risk: High decision consequence (batch disposition) but low model influence due to human-in-the-loop QA sign-off.", "§III.A, §III.G–H"],
        ["4", "Develop Credibility Plan", "Frozen stratified splits; multi-seed protocol; declared baseline; pre-specified faithfulness metrics incl. random control.", "§IV.B–D"],
        ["5", "Execute the Plan", "Multi-seed benchmark (Table III); paired significance tests (§V.B); per-class bootstrap CIs (Table IV); faithfulness vs. control (Table V).", "§V"],
        ["6", "Document Results & Deviations", "Negative statistical result disclosed (p = 0.513); training capacity asymmetry disclosed; proxy-domain substitution reported.", "§IV.A, §IV.C, §V.B, §VII"],
        ["7", "Determine Model Adequacy for COU", "Adequate as feasibility evidence; not yet adequate for cGMP release. §VII documents remaining gaps.", "§VII"],
        ["—", "Lifecycle Management", "Append-only audit schema; image/model/split hashing; site transfer-calibration protocol.", "§III.F, §VII"],
    ]
    t6 = doc.add_table(rows=len(t6_rows)+1, cols=len(t6_headers))
    for c_idx, h in enumerate(t6_headers):
        p_c = t6.rows[0].cells[c_idx].paragraphs[0]
        p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p_c.add_run(h)
        format_run(r, "Times New Roman", 8, bold=True)
    for r_idx, row_data in enumerate(t6_rows):
        for c_idx, val in enumerate(row_data):
            p_c = t6.rows[r_idx+1].cells[c_idx].paragraphs[0]
            p_c.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 2 else WD_ALIGN_PARAGRAPH.CENTER
            r = p_c.add_run(val)
            format_run(r, "Times New Roman", 7.5)
    t6_widths = [Inches(0.40), Inches(1.80), Inches(3.85), Inches(1.10)]
    set_table_booktabs(t6, t6_widths)
    add_p(doc, "", space_before=0, space_after=2)

    # Resume 2 columns
    s_b3 = doc.add_section(WD_SECTION.CONTINUOUS)
    add_columns_to_section(s_b3, num_cols=2, space_pt=18)

    # VII. LIMITATIONS
    add_heading_1(doc, "VII. LIMITATIONS")
    limits = [
        ("1. Proxy domain: ", "MVTec AD Bottle captures optical geometry but its defects are structural analogies, not physical freeze-drying failure modes."),
        ("2. Sample scale & power: ", "44 test images (3 per defect class), 9 defect masks, and 3 seeds over 1 frozen split. This is powered as a feasibility study."),
        ("3. Non-capacity-matched comparison: ", "ViT-B/16 trained 16.7% parameters vs ResNet-50's 100% due to a 6 GB VRAM budget; deployment-realistic rather than capacity-matched."),
        ("4. Single frozen split: ", "All runs share one split; reported variance reflects initialization and augmentation rather than partition variance."),
        ("5. Checkpoint selection weight: ", "Both architectures memorize the training set quickly; validation ranking is inverted relative to test ranking, indicating selection noise."),
        ("6. Bounded faithfulness: ", "Rollout and Grad-CAM agree moderately (IoU 0.466); Rollout localizes poorly against masks. Dual presentation is necessary."),
        ("7. Unvalidated disposition rule: ", "Weights in (14) are specified by design; human-factors calibration with qualified reviewers remains future work."),
        ("8. Inference latency vs. line takt: ", "Inference is 18 ms but generating both heatmaps takes ~45 ms. Two-tier inline/offline screening is required for 400–600 vials/min."),
        ("9. Single optical setup: ", "Evaluation utilized standardized illumination; facility variations in fill volume and glass thickness require transfer calibration."),
        ("10. Statistical reproducibility: ", "PyTorch GPU operations are statistically reproducible across seeds, but cuDNN non-determinism prevents bitwise matching."),
    ]
    for l_h, l_t in limits:
        p_l = doc.add_paragraph()
        p_l.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_l.paragraph_format.left_indent = Inches(0.12)
        p_l.paragraph_format.space_before = Pt(1)
        p_l.paragraph_format.space_after = Pt(2)
        r_h = p_l.add_run(l_h)
        format_run(r_h, "Times New Roman", 9.5, bold=True)
        r_t = p_l.add_run(l_t)
        format_run(r_t, "Times New Roman", 9.5)

    # VIII. CONCLUSION
    add_heading_1(doc, "VIII. CONCLUSION AND FUTURE WORK")
    add_p(doc, "This paper presented ViT-XAI-Lyo, an attention-explainable Vision Transformer framework for lyophilized cake defect classification "
               "with regulatory-aligned interpretability. On an industrial proxy benchmark, ViT-B/16 matched a fully fine-tuned ResNet-50 baseline "
               "(macro-F1 0.843 ± 0.100 vs 0.792 ± 0.015; p = 0.513 seed-level, p = 0.542 image bootstrap) while updating only 16.7% of its parameters. "
               "Quantitative faithfulness evaluation against pixel-level masks and a random control isolated Grad-CAM as the faithful localization method "
               "and Rollout as providing global context. We established an audit-metadata schema, a batch-disposition risk score, and an FDA crosswalk.\n\n"
               "Future work will focus on: (i) acquiring multi-view commercial vial imagery; (ii) capacity-matched training on large GPUs; "
               "(iii) gradient-weighted attention rollout (GMAR [15]); (iv) a human-factors study to validate disposition escalation; and "
               "(v) model distillation for ultra-fast line-speed deployment.")

    # DATA AVAILABILITY
    add_heading_1(doc, "DATA AND CODE AVAILABILITY")
    add_p(doc, "Source code, data preparation scripts, frozen splits (metadata.csv), metric JSON logs, and the Gradio demonstration interface are available "
               "from the corresponding author on request and published at https://github.com/sit24am-projects/ViT-XAI-Lyo.")

    # REFERENCES
    add_heading_1(doc, "REFERENCES")
    refs = [
        "[1] F. Franks, \"Freeze-drying of bioproducts: putting principles into practice,\" European Journal of Pharmaceutics and Biopharmaceutics, vol. 45, no. 3, pp. 221–229, 1998.",
        "[2] X. Tang and M. J. Pikal, \"Design of freeze-drying processes for pharmaceuticals: Practical advice,\" Pharmaceutical Research, vol. 21, no. 2, pp. 191–200, 2004.",
        "[3] M. J. Pikal and S. Shah, \"The collapse temperature in freeze drying: Dependence on measurement methodology and rate of water removal from the glassy phase,\" International Journal of Pharmaceutics, vol. 62, no. 2–3, pp. 165–186, 1990.",
        "[4] B. S. Chang and S. Yeung, \"Physical stability of protein pharmaceuticals,\" in Formulation and Process Development Strategies for Manufacturing Biopharmaceuticals, Wiley, pp. 69–104, 2010.",
        "[5] S. M. Patel, T. Doen, and M. J. Pikal, \"Determination of end point of primary drying in freeze-drying process control,\" AAPS PharmSciTech, vol. 11, no. 1, pp. 73–84, 2010.",
        "[6] Z. Li and C. Tsay, \"Defect detection in lyophilized drug products with convolutional neural networks,\" U.S. Patent 11,263,738 B2 (assignee: Genentech, Inc.), Mar. 1, 2022.",
        "[7] Z. Li and C. Tsay, \"Defect detection in lyophilized drug products with convolutional neural networks,\" U.S. Patent 11,810,287 B2 (assignee: Genentech, Inc.), Nov. 7, 2023.",
        "[8] D. Jiang, K. Zhu, N. Tao, and X. Ren, \"An improved YOLOv11-based method for defect detection in lyophilized vial,\" Electronics, vol. 14, no. 22, art. 4526, 2025.",
        "[9] H. Xu, Q. Liu, J. Zhu, H. Dai, and D. Zhang, \"CSLNet: An enhanced YOLOv8-based approach to defect surface foreign objects in lyophilized powder,\" Signal, Image and Video Processing, vol. 19, no. 9, art. 728, 2025.",
        "[10] P. Müller, A. Sack, J. Dümler, M. Heckel, T. Wenzel, T. Siegert, et al., \"Automated tomographic assessment of structural defects of freeze-dried pharmaceuticals,\" AAPS PharmSciTech, vol. 25, no. 6, art. 143, 2024.",
        "[11] U.S. Food and Drug Administration, \"Electronic records; electronic signatures,\" 21 CFR Part 11, Washington, DC, USA.",
        "[12] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, et al., \"An image is worth 16x16 words: Transformers for image recognition at scale,\" in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.",
        "[13] K. Han, Y. Wang, H. Chen, X. Chen, J. Guo, Z. Liu, et al., \"A survey on vision transformer,\" IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 1, pp. 87–110, 2023.",
        "[14] S. Abnar and W. Zuidema, \"Quantifying attention flow in transformers,\" in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), pp. 4190–4197, 2020.",
        "[15] S. Jo, G. Jang, and H. Park, \"GMAR: Gradient-driven multi-head attention rollout for vision transformer interpretability,\" arXiv:2504.19414, 2025.",
        "[16] V. Petsiuk, A. Das, and K. Saenko, \"RISE: Randomized input sampling for explanation of black-box models,\" in Proc. British Machine Vision Conf. (BMVC), 2018.",
        "[17] U.S. Food and Drug Administration, CDER, \"Considerations for the use of artificial intelligence to support regulatory decision-making for drug and biological products,\" Draft Guidance for Industry, Docket FDA-2024-D-4689, Jan. 2025.",
        "[18] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, \"Grad-CAM: Visual explanations from deep networks via gradient-based localization,\" in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), pp. 618–626, 2017.",
        "[19] H. Xu, Y. Xu, and K. Hu, \"A vision-based inspection system for pharmaceutical production line,\" Journal of Supercomputing, vol. 81, no. 4, art. 625, 2025.",
        "[20] S. Li, C. Wu, and N. Xiong, \"Hybrid architecture based on CNN and transformer for strip steel surface defect classification,\" Electronics, vol. 11, no. 8, art. 1200, 2022.",
        "[21] K. Parmar, R. Jain, P. Anitha, A. Jayanthi, R. Putchanuthala, et al., \"A misclassification-aware explainable hybrid CNN–vision transformer framework for radiographic weld inspection,\" Scientific Reports, vol. 16, no. 1, art. 17134, 2026.",
        "[22] S. Kahng and H. Kim, \"Self-supervised representation learning for wafer bin map defect pattern classification,\" IEEE Transactions on Semiconductor Manufacturing, vol. 34, no. 1, pp. 74–86, 2021.",
        "[23] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, \"MVTec AD — A comprehensive real-world dataset for unsupervised anomaly detection,\" in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 9584–9592, 2019."
    ]
    for ref_text in refs:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_ref.paragraph_format.left_indent = Inches(0.22)
        p_ref.paragraph_format.first_line_indent = Inches(-0.22)
        p_ref.paragraph_format.space_before = Pt(0)
        p_ref.paragraph_format.space_after = Pt(2.5)
        p_ref.paragraph_format.line_spacing = 1.0
        r_ref = p_ref.add_run(ref_text)
        format_run(r_ref, "Times New Roman", 8)

    # Balance 2 columns on final page
    doc.add_section(WD_SECTION.CONTINUOUS)

    doc.save(DOCX_OUT)
    print(f"[OK] Saved Word Document -> {DOCX_OUT.resolve()}")
    
    # Save a duplicate copy to .artifacts/submission-readiness/
    art_docx = ARTIFACTS_DIR / DOCX_OUT.name
    doc.save(art_docx)
    print(f"[OK] Saved Artifact Copy -> {art_docx.resolve()}")

def export_docx_to_pdf():
    import win32com.client
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        in_path = str(DOCX_OUT.resolve())
        out_path = str(PDF_OUT.resolve())
        doc_obj = word.Documents.Open(in_path)
        doc_obj.SaveAs(out_path, FileFormat=17) # 17 = wdFormatPDF
        doc_obj.Close()
        print(f"[OK] Exported PDF -> {out_path}")
        
        # Save a duplicate copy to .artifacts/submission-readiness/
        art_pdf = ARTIFACTS_DIR / PDF_OUT.name
        import shutil
        shutil.copyfile(out_path, str(art_pdf.resolve()))
        print(f"[OK] Saved Artifact Copy -> {art_pdf.resolve()}")
    finally:
        word.Quit()

if __name__ == "__main__":
    build_paper_document()
    export_docx_to_pdf()
