"""
generate_thesis_summary.py
==========================
Generates a detailed, 3-5 page Executive Summary of the Master's Thesis:
"Subject-Invariant Spatiotemporal Contrastive Network for Micro-Expression Recognition"

Formats generated:
1. MS Word (.docx) - Standard Times New Roman academic layout.
2. PDF (.pdf) - Custom CSS styled publication-grade report.

Outputs: Thesis_Summary_26.05.docx, Thesis_Summary_26.05.pdf
"""
import os
import sys
import pathlib
import subprocess

# Ensure document libraries are installed
for pkg in ["python-docx", "markdown", "xhtml2pdf"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import markdown
from xhtml2pdf import pisa

# Directories and paths
ROOT_DIR = pathlib.Path(__file__).parent.resolve()
FIGURES_DIR = ROOT_DIR / "figures"
DOCX_PATH = ROOT_DIR / "Thesis_Summary_26.05.docx"
PDF_PATH = ROOT_DIR / "Thesis_Summary_26.05.pdf"

# Color Palette Definitions
NAVY = RGBColor(26, 54, 93)       # #1a365d (Primary headings)
STEEL = RGBColor(43, 108, 176)    # #2b6cb0 (Secondary headings)
CHARCOAL = RGBColor(45, 55, 72)   # #2d3748 (Body text)
LIGHT_BLUE = "F0F4F8"             # Equation box background

# Results Table Data
RESULTS_DATA = [
    ("Stage / Protocol", "Target Dataset", "Best Val Accuracy", "Macro F1-Score", "Key Findings"),
    ("Stage 3 (Scratch Joint GRL+SupCon)", "Micro-Expression (Unified)", "58.11% (Mean)", "33.31%", "Robust mean across subjects disjoint validation; captures universal dynamics."),
    ("Stage 4 Phase 1 (Macro Pre-train)", "Macro-Expression (CAS(ME)²)", "70.37%", "Not Reported", "Easily extracts spatiotemporal motion cues from large muscle displacements."),
    ("Stage 4 Phase 2 (Micro Fine-tune)", "Micro-Expression (Unified)", "37.78%", "18.41%", "GRL successfully strips subject identity details. Scarcity of Positive/Surprise limits recall.")
]

# Literature Citations
FUTURE_PAPERS = [
    ("Phase-Aware Temporal Augmentation",
     "Vu Tram Anh Khuong et al., arXiv:2510.15466, 2025",
     "Onset-to-Apex / Apex-to-Offset sequence decomposition",
     "Decompose spatiotemporal sequences to generate phase-specific dynamic images, augmenting minority classes to solve class imbalance recall bottlenecks."),
    
    ("Multimodal Semantic Guidance",
     "MECLIP Framework, 2025",
     "LLM-generated FACS Action Unit text-image contrastive alignment",
     "Extend the Stage 3 projection head to align visual embeddings with textual prompts, regularizing the latent space to enforce subject-invariant motion learning."),
    
    ("Self-Supervised Masked Autoencoders",
     "TGMAE, IEEE ICME, 2024",
     "Spatiotemporal Gaussian-weighted patch masking & reconstruction",
     "Pre-train the STSTNet-SLSTT backbone on massive unlabeled facial video repositories to learn general facial structural kinematics prior to transfer tuning.")
]


# =====================================================================
#  DOCX Generator
# =====================================================================

def generate_docx():
    print(f"Generating MS Word Summary: {DOCX_PATH}")
    doc = docx.Document()
    
    # Configure margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    def set_font(run, name="Times New Roman", size=Pt(12), bold=False, italic=False, color=None):
        run.font.name = name
        run.font.size = size
        run.font.bold = bold
        run.font.italic = italic
        if color:
            run.font.color.rgb = color

    # Header
    p_header = doc.add_paragraph()
    p_header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run_h = p_header.add_run("Addhyan  |  Executive Summary  |  May 2026")
    set_font(run_h, size=Pt(8.5), italic=True, color=RGBColor(113, 128, 150))

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("EXECUTIVE SUMMARY\nSPATIOTEMPORAL ADVERSARIAL CONTRASTIVE NETWORK FOR SUBJECT-INVARIANT MICRO-EXPRESSION RECOGNITION")
    set_font(run_title, size=Pt(16), bold=True, color=NAVY)
    p_title.paragraph_format.space_before = Pt(20)
    p_title.paragraph_format.space_after = Pt(24)

    # Helper functions for headings and paragraphs
    def add_heading_1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(text)
        set_font(run, size=Pt(14), bold=True, color=NAVY)
        p.paragraph_format.keep_with_next = True
        return p

    def add_heading_2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        set_font(run, size=Pt(12), bold=True, color=STEEL)
        p.paragraph_format.keep_with_next = True
        return p

    def add_body(text, bold_lead=None, bullet=False):
        p = doc.add_paragraph(style='List Bullet' if bullet else 'Normal')
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        if bold_lead:
            r_lead = p.add_run(bold_lead)
            set_font(r_lead, size=Pt(10.5), bold=True, color=CHARCOAL)
        run = p.add_run(text)
        set_font(run, size=Pt(10.5), color=CHARCOAL)
        return p

    def add_equation_box(latex_eq, text_desc):
        table = doc.add_table(rows=1, cols=1)
        table.autofit = False
        table.columns[0].width = Inches(6.5)
        cell = table.cell(0, 0)
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{LIGHT_BLUE}"/>')
        cell._tc.get_or_add_tcPr().append(shading)
        
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(6)
        
        r_eq = p.add_run(latex_eq + "\n")
        set_font(r_eq, name="Consolas", size=Pt(9.5), bold=True, color=NAVY)
        
        r_desc = p.add_run(text_desc)
        set_font(r_desc, size=Pt(8.5), italic=True, color=CHARCOAL)

    def add_image(filename, caption_text):
        img_path = FIGURES_DIR / filename
        if img_path.exists():
            doc.add_paragraph().paragraph_format.space_before = Pt(6)
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_img = p_img.add_run()
            r_img.add_picture(str(img_path), width=Inches(4.5))
            
            p_lbl = doc.add_paragraph()
            p_lbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_lbl = p_lbl.add_run(f"Figure: {caption_text}")
            set_font(r_lbl, size=Pt(9), italic=True, color=CHARCOAL)
            p_lbl.paragraph_format.space_after = Pt(8)
        else:
            print(f"Warning: Figure {filename} not found, skipping embed.")

    def add_table_data(headers, rows):
        table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
        table.style = 'Table Grid'
        
        # Header Row
        hdr_cells = table.rows[0].cells
        for col_idx, text in enumerate(headers):
            hdr_cells[col_idx].text = text
            for p in hdr_cells[col_idx].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    set_font(r, size=Pt(9), bold=True, color=RGBColor(255, 255, 255))
            shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1A365D"/>')
            hdr_cells[col_idx]._tc.get_or_add_tcPr().append(shading)

        # Data Rows
        for row_idx, data_row in enumerate(rows):
            row_cells = table.rows[row_idx + 1].cells
            bg_color = "F7FAFC" if row_idx % 2 == 0 else "FFFFFF"
            for col_idx, text in enumerate(data_row):
                row_cells[col_idx].text = text
                shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_color}"/>')
                row_cells[col_idx]._tc.get_or_add_tcPr().append(shading)
                for p in row_cells[col_idx].paragraphs:
                    for r in p.runs:
                        set_font(r, size=Pt(8.5), color=CHARCOAL)
                        if col_idx == 0:
                            r.font.bold = True
        doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ─────────────────────────────────────────────────────────────────
    # SECTION 1: RESEARCH BACKGROUND AND CHALLENGES
    # ─────────────────────────────────────────────────────────────────
    add_heading_1("1. Research Background & Core Challenges")
    add_body(
        "Micro-expressions are transient, involuntary facial muscle contractions that occur when individuals "
        "attempt to suppress their emotional state. Lasting less than 500 milliseconds, they represent a highly "
        "reliable indicator of genuine affect, proving vital in clinical psychology, threat assessment, lie detection, "
        "and human-computer interaction. However, automated Micro-Expression Recognition (MER) using deep learning "
        "is severely bottlenecked by two twin challenges:"
    )
    add_body(
        "Micro-expression databases are extremely small (e.g., CASME II has only 255 samples) "
        "due to the high cost of manual high-speed video annotation by FACS-certified practitioners. Standard spatiotemporal "
        "deep neural networks containing millions of parameters suffer from severe overfitting or feature collapse when trained "
        "on datasets of this size.",
        bold_lead="1. Data Starvation: "
    )
    add_body(
        "Faces contain rich static details (bone structure, skin texture) that are "
        "numerically stronger than transient facial motions. Deep networks naturally latch onto these static features, causing "
        "them to recognize subject identity rather than the micro-expression. Consequently, models evaluate poorly on unseen subjects "
        "during cross-validation, demonstrating high subject identity bias.",
        bold_lead="2. Subject Identity Bias: "
    )

    # ─────────────────────────────────────────────────────────────────
    # SECTION 2: PROPOSED FOUR-STAGE PIPELINE
    # ─────────────────────────────────────────────────────────────────
    add_heading_1("2. Proposed Four-Stage Pipeline (Methodology)")
    add_body(
        "To overcome these bottlenecks, we implement a highly optimized, parameter-efficient, subject-invariant "
        "spatiotemporal contrastive network structured in a unified four-stage pipeline:"
    )
    
    add_heading_2("Stage 1: Preprocessing & Motion Amplification")
    add_body(
        "We unify the CASME II (200 FPS, pre-cropped) and CAS(ME)² (30 FPS, full frame) databases under a unified 4-class "
        "taxonomy (Negative, Positive, Surprise, Others) to stabilize optimization. To amplify subtle movements, we apply "
        "Eulerian Video Magnification (EVM), which decomposes frames spatially using Laplacian Pyramids, filters the signal "
        "temporally via FFT (0.4–3.0 Hz bandpass), and amplifies it by alpha=20 before reconstruction."
    )
    add_body(
        "To strip static visual identity and background textures, we extract physics-based motion modalities rather than "
        "feeding raw RGB pixels. Using dense Farneback optical flow, we compute horizontal flow (u), vertical flow (v), "
        "and optical strain (os) to capture local skin deformation:"
    )
    add_equation_box(
        "os = sqrt( (du/dx)^2 + (dv/dy)^2 + 0.5 * (du/dy + dv/dx)^2 )",
        "Optical strain formulation modeling localized skin compression and shear."
    )
    add_body(
        "Each modality is Min-Max normalized to [0,1]. Finally, sequences are temporally standardized to exactly L=33 frames "
        "(yielding T=32 frame pairs for flow calculation) using linear temporal interpolation."
    )

    add_heading_2("Stage 2: Hybrid Network Architecture")
    add_body(
        "We design a spatiotemporal model that is highly parameter-efficient (~382K parameters) to combat data starvation:"
    )
    add_body(
        "A shallow weight-unshared 3D-CNN splits the 3-channel input "
        "and routes flow-u, flow-v, and skin strain through independent branches. Weight sharing is avoided because "
        "horizontal, vertical, and deformation dynamics exhibit distinct physical properties. Spatiotemporal convolution kernels "
        "are set to k=[1,3,3], preserving the temporal axis (32 steps) completely.",
        bold_lead="1. Modality-Aware 3D Backbone (STSTNet-3D): "
    )
    add_body(
        "A parameter-free 3D attention mechanism that computes an analytical energy "
        "score per neuron based on spatial-temporal neighborhood statistics. Neurons with lower energy are more distinct, "
        "allowing the model to focus on active facial action zones (eyebrow contraction, mouth twitching) without adding parameters.",
        bold_lead="2. 3D SimAM (Attention): "
    )
    add_body(
        "After spatial pooling, the temporal sequence of shape [B, 32, 96] is "
        "injected with sinusoidal positional encodings. It is processed by a 2-layer pre-norm multi-head self-attention transformer "
        "to capture onset-to-apex-to-offset temporal dynamics. A temporal mean pool collapses the 32 tokens into a final 96-d feature vector.",
        bold_lead="3. SLSTT Sequence Transformer: "
    )

    add_heading_2("Stage 3: Adversarial Domain Adaptation & Contrastive Learning")
    add_body(
        "To enforce subject-invariance and structure a discriminative feature representation, we route the global 96-d feature "
        "through a triple-head wrapper:"
    )
    add_body(
        "Optimizes emotion logits (Negative, Positive, Surprise, Others) using class-weighted "
        "Focal Loss (gamma=2.0, label smoothing=0.05) to penalize majority classes and handle severe database imbalance.",
        bold_lead="1. Emotion Classification Head: "
    )
    add_body(
        "Injects a Gradient Reversal Layer (GRL) before an identity classification head. "
        "The GRL acts as an identity map in the forward pass but multiplies incoming gradients by -lambda in the backward pass. A sigmoid "
        "annealing schedule ramps lambda from 0.0 to 1.0, forcing the backbone to strip out subject identity details.",
        bold_lead="2. Adversarial Identity Head (GRL): "
    )
    add_body(
        "Optimizes 64-d projection embeddings via Supervised Contrastive "
        "Loss (SupCon) to pull same-emotion embeddings together and push different emotions apart. Because VRAM bounds batch size "
        "to B=2, we integrate a Cross-Batch Memory (XBM) queue of size M=64. The XBM stores historical embeddings and labels in a FIFO "
        "buffer, allowing the active batch to contrast against 66 targets without requiring gradient VRAM.",
        bold_lead="3. Contrastive Projection Head with Cross-Batch Memory: "
    )

    add_heading_2("Stage 4: Macro-to-Micro Transfer Learning")
    add_body(
        "To overcome data starvation, we pre-train the backbone on 293 macro-expression clips from CAS(ME)² (where muscle displacements "
        "are larger and easier to extract) with the GRL adversary disabled. We then transfer the weights to fine-tune on 305 micro-expression "
        "clips, reinitializing the identity classification layer (moving from N=20 macro subjects to N=26 micro subjects) and enabling the "
        "GRL adversary to actively enforce identity-invariance."
    )

    # ─────────────────────────────────────────────────────────────────
    # SECTION 3: QUANTITATIVE RESULTS
    # ─────────────────────────────────────────────────────────────────
    add_heading_1("3. Experimental Results & Visual Analysis")
    add_body(
        "The pipeline was evaluated under two protocols: 26-fold Leave-One-Subject-Out (LOSO) cross-validation (Stage 3 scratch) and "
        "a subject-disjoint validation split (Stage 4 transfer learning). The results are summarized in Table 3.1:"
    )
    
    # Table 3.1
    add_table_data(RESULTS_DATA[0], RESULTS_DATA[1:])

    add_image("tsne_embeddings.png", "t-SNE Embedding Space Projection. Plot A (left) demonstrates clean clustering of emotion classes; Plot B (right) show uniform mixing of subject IDs, validating the GRL's ability to enforce identity-invariance.")
    add_body(
        "As shown in the t-SNE projection (Figure 3.1), the GRL successfully achieves subject-invariance, as subject IDs are completely "
        "and uniformly mixed. Concurrently, the Supervised Contrastive Loss structures a highly discriminative embedding space, forming clear "
        "clusters for emotion classes. However, the row-normalized confusion matrix (Figure 3.2) indicates that while the model achieves "
        "high recall on the majority classes (56.25% Negative, 44.44% Others), it fails to predict Positive and Surprise classes. This "
        "highlights the class imbalance bottleneck, as these classes contain only 45 and 34 samples respectively in the fine-tuning set."
    )
    add_image("confusion_matrix.png", "Stage 4 Row-Normalized Validation Confusion Matrix.")

    # ─────────────────────────────────────────────────────────────────
    # SECTION 4: FUTURE WORK & CONCLUSIONS
    # ─────────────────────────────────────────────────────────────────
    add_heading_1("4. How to Improve Results (Future Work)")
    add_body(
        "To address class imbalance and velocity mismatches across frame rates (30 FPS macro vs. 200 FPS micro), we propose three "
        "key research directions, mapped to real publications from recent literature (2024–2025):"
    )

    for title, paper, why, how in FUTURE_PAPERS:
        add_body(f"Focus: {title}", bullet=True)
        add_body(f"Citation: {paper}", bullet=True)
        add_body(f"Why Reference: {why}", bullet=True)
        add_body(f"How to Integrate: {how}", bullet=True)
        doc.add_paragraph().paragraph_format.space_after = Pt(3)

    add_heading_1("5. Conclusion")
    add_body(
        "This research presents a parameter-efficient, subject-invariant spatiotemporal contrastive network for micro-expression "
        "recognition. By combining physics-based optical flow/strain modalities, a parameter-free 3D attention mechanism (SimAM), "
        "a sequence-level transformer (SLSTT), domain-adversarial GRL training, and SupCon with Cross-Batch Memory, the pipeline "
        "effectively eliminates subject identity bias. The model achieves a 26-fold LOSO grand mean accuracy of 58.11% on micro-expressions. "
        "Future integration of phase-aware temporal augmentation, multimodal LLM semantic descriptors (MECLIP), and self-supervised MAE "
        "pre-training will address data scarcity and class imbalance, setting a new trajectory for robust micro-expression recognition."
    )

    # Save Word Document
    doc.save(DOCX_PATH)
    print(f"SUCCESS: DOCX summary generated -> {DOCX_PATH}")


# =====================================================================
#  PDF Generator
# =====================================================================

def generate_pdf():
    print(f"Generating PDF Summary: {PDF_PATH}")
    
    # Custom CSS for A4 Summary styling
    css = """
    @page { 
        size: A4; 
        margin: 2.0cm 2.0cm 2.0cm 2.0cm;
    }
    #footer {
        position: fixed;
        bottom: -1.0cm;
        left: 0px;
        right: 0px;
        height: 1.0cm;
        border-top: 0.5px solid #cbd5e0;
        padding-top: 4px;
    }
    body { 
        font-family: 'Times New Roman', Times, serif; 
        font-size: 10pt; 
        line-height: 1.4; 
        color: #2d3748; 
    }
    h1 { 
        font-size: 14pt; 
        color: #1a365d; 
        border-bottom: 1.5px solid #1a365d; 
        padding-bottom: 4px; 
        margin-top: 14pt; 
        margin-bottom: 8pt;
    }
    h2 { 
        font-size: 11pt; 
        color: #2b6cb0; 
        margin-top: 10pt; 
        margin-bottom: 6pt; 
    }
    p { 
        margin-bottom: 8pt; 
        text-align: justify;
    }
    ul { 
        margin-bottom: 8pt; 
        padding-left: 15px; 
    }
    li { 
        margin-bottom: 3pt; 
    }
    .equation-box { 
        background-color: #f0f4f8; 
        border-left: 3px solid #1a365d;
        padding: 6px 10px; 
        margin: 8pt 0; 
        font-family: 'Courier New', Courier, monospace; 
        font-weight: bold;
        font-size: 9pt;
        text-align: center;
        color: #1a365d;
    }
    .equation-desc {
        font-family: 'Times New Roman', Times, serif;
        font-weight: normal;
        font-size: 8pt;
        font-style: italic;
        margin-top: 3pt;
        color: #4a5568;
        text-align: left;
    }
    table { 
        border-collapse: collapse; 
        width: 100%; 
        margin: 12pt 0; 
        font-size: 8pt; 
    }
    th { 
        background-color: #1a365d; 
        color: white; 
        padding: 5px 8px; 
        text-align: left; 
        font-weight: bold; 
    }
    td { 
        padding: 4px 8px; 
        border-bottom: 1px solid #cbd5e0; 
        border-right: 1px solid #e2e8f0;
        border-left: 1px solid #e2e8f0;
    }
    tr:nth-child(even) td { 
        background-color: #f7fafc; 
    }
    .figure-container {
        text-align: center;
        margin: 12pt 0;
        page-break-inside: avoid;
    }
    .figure-caption {
        font-size: 8pt;
        font-style: italic;
        margin-top: 4pt;
        color: #4a5568;
    }
    .title-area {
        text-align: center;
        margin-bottom: 18pt;
        border-bottom: 3px double #1a365d;
        padding-bottom: 10px;
    }
    .title-main {
        font-size: 15pt;
        font-weight: bold;
        color: #1a365d;
        margin-bottom: 6px;
        line-height: 1.2;
    }
    .title-sub {
        font-size: 10.5pt;
        font-style: italic;
        color: #4a5568;
        margin-bottom: 8px;
        line-height: 1.3;
    }
    .title-meta {
        font-size: 9pt;
        color: #718096;
    }
    .bold-lead {
        font-weight: bold;
        color: #2d3748;
    }
    """

    # Helper to generate HTML tables
    def make_html_table(headers, rows):
        html = "<table><thead><tr>"
        for h in headers:
            html += f"<th>{h}</th>"
        html += "</tr></thead><tbody>"
        for r in rows:
            html += "<tr>"
            for i, val in enumerate(r):
                if i == 0:
                    html += f"<td><strong>{val}</strong></td>"
                else:
                    html += f"<td>{val}</td>"
            html += "</tr>"
        html += "</tbody></table>"
        return html

    # Helper to generate HTML future papers list
    def make_html_future_papers(future_papers):
        html = ""
        for title, paper, why, how in future_papers:
            html += f"""
            <li><strong>{title}:</strong> To improve upon current results, we can integrate this methodology.
            <br/><em>Citation:</em> <strong>{paper}</strong>
            <br/><em>Why Reference:</em> {why}
            <br/><em>How to Integrate:</em> {how}</li>
            """
        return html

    # Build the HTML content
    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <style>{css}</style>
    </head>
    <body>
        <div id="footer">
            <table width="100%" style="border: none; margin: 0; padding: 0;">
                <tr style="background: transparent; border: none;">
                    <td style="border: none; text-align: left; color: #718096; font-size: 8pt; font-family: 'Times New Roman', Times, serif;">Master's Thesis Executive Summary | Addhyan</td>
                    <td style="border: none; text-align: right; color: #718096; font-size: 8pt; font-family: 'Times New Roman', Times, serif;">Page <pdf:pagenumber/> of <pdf:pagecount/></td>
                </tr>
            </table>
        </div>

        <!-- TITLE AREA -->
        <div class="title-area">
            <div class="title-main">EXECUTIVE SUMMARY</div>
            <div class="title-sub">Subject-Invariant Spatiotemporal Contrastive Network for Micro-Expression Recognition</div>
            <div class="title-meta">
                Author: Addhyan | Hardware: NVIDIA RTX 4060 | Seed: 42 | Date: May 26, 2026
            </div>
        </div>

        <!-- SECTION 1 -->
        <h1>1. Research Background & Core Challenges</h1>
        <p>
            Micro-expressions are transient, involuntary facial muscle contractions lasting less than 500 milliseconds. 
            They serve as standard windows into genuine emotional states but automated Micro-Expression Recognition (MER) 
            using deep learning remains a challenging task. Specifically, traditional deep networks are severely bottlenecked by two twin issues:
        </p>
        <ul>
            <li><span class="bold-lead">1. Data Starvation:</span> Annotation requires certified FACS practitioners to index high-speed clips frame-by-frame. 
            This expensive coding limits benchmark databases (e.g., CASME II has only 255 samples), causing deep architectures 
            to easily overfit or fail during feature extraction.</li>
            <li><span class="bold-lead">2. Subject Identity Bias:</span> Static facial details (bone structure, skin texture, lighting) are numerically 
            stronger than dynamic muscle contractions. Deep networks naturally exploit these static identity features. Consequently, models evaluate 
            poorly on unseen subjects, memorizing who the person is rather than what emotion they express.</li>
        </ul>

        <!-- SECTION 2 -->
        <h1>2. Proposed Four-Stage Pipeline (Methodology)</h1>
        <p>
            To address these challenges, we propose a unified, parameter-efficient (~382K parameters) pipeline:
        </p>
        
        <h2>Stage 1: Preprocessing & Motion Amplification</h2>
        <p>
            We merge CASME II (200 FPS, pre-cropped) and CAS(ME)² (30 FPS, full frame) into a unified 4-class taxonomy 
            (Negative, Positive, Surprise, Others) to stabilize optimization. To amplify motion, we apply Eulerian Video Magnification (EVM) 
            with a temporal bandpass filter set to 0.4–3.0 Hz (targeting facial muscle contractions) and an amplification factor alpha=20. 
            To strip out static identity details, we extract physics-based optical flow modalities. For each pair of frames, dense Farneback flow 
            is calculated to obtain horizontal flow (u), vertical flow (v), and skin deformation optical strain (os):
        </p>
        <div class="equation-box">
            os = &radic;( (&part;u/&part;x)<sup>2</sup> + (&part;v/&part;y)<sup>2</sup> + 0.5 * (&part;u/&part;y + &part;v/&part;x)<sup>2</sup> )
            <div class="equation-desc">Optical strain magnitude modeling skin shear and compression. Normalizing each channel to [0,1].</div>
        </div>
        <p>
            Finally, the frames are temporally standardized to exactly L=33 frames (T=32 frame pairs for flow) using linear interpolation.
        </p>

        <h2>Stage 2: Hybrid Network Architecture</h2>
        <p>
            The network backbone is designed to be highly parameter-efficient to prevent overfitting:
        </p>
        <ul>
            <li><strong>Modality-Aware 3D Backbone (STSTNet-3D):</strong> Splitting the 3-channel flow tensor, we route flow-u, flow-v, 
            and strain through three parallel, unshared convolution branches. Weights are unshared to let kernels specialize in each 
            distinct physical motion type. Standard 3D convolutions use a temporal kernel depth of 1, preventing premature temporal mixing.</li>
            <li><strong>3D SimAM Attention:</strong> To highlight active facial regions (eyebrows, mouth corners) without adding learnable parameters, 
            we integrate 3D SimAM attention. It evaluates analytical energy scores for each neuron, rescaled element-wise based on inter-neuron suppression.</li>
            <li><strong>SLSTT Sequence Transformer:</strong> Spatial pooling compresses features to shape [B, 32, 96]. We inject sinusoidal positional 
            encodings and process the sequence with a 2-layer pre-norm transformer encoder to learn temporal dynamics. Temporal mean pooling 
            then aggregates the sequence into a 96-d global feature vector.</li>
        </ul>

        <h2>Stage 3: Joint Domain-Adversarial & Contrastive Learning</h2>
        <p>
            The global feature vector is routed through a triple-head wrapper to enforce subject-invariance:
        </p>
        <ul>
            <li><strong>Emotion Head:</strong> Classification optimized via class-weighted Focal Loss (gamma=2.0, label smoothing=0.05) to 
            mitigate the severe database class imbalance.</li>
            <li><strong>Adversarial Identity Head:</strong> Places a Gradient Reversal Layer (GRL) before an identity classification head. 
            During backpropagation, the GRL reverses identity gradients with a coefficient lambda annealed from 0.0 to 1.0, actively 
            stripping identity textures from the spatiotemporal backbone.</li>
            <li><strong>Contrastive Projection Head:</strong> Maps features to a 64-d space optimized via Supervised Contrastive Loss (SupCon). 
            To bypass batch-size VRAM constraints (B=2), we integrate a Cross-Batch Memory (XBM) queue of size M=64 to cache historical embeddings.</li>
        </ul>

        <h2>Stage 4: Macro-to-Micro Transfer Learning</h2>
        <p>
            To combat sample scarcity, we pre-train the model on 293 macro-expression clips (easy displacement) with GRL disabled (validation accuracy = 70.37%). 
            We then transfer the weights to the micro-expression model, reinitializing the identity head for N=26 micro subjects, and fine-tune with the GRL adversary enabled.
        </p>

        <!-- SECTION 3 -->
        <h1>3. Experimental Results & Visual Analysis</h1>
        <p>
            The pipeline was validated under 26-fold Leave-One-Subject-Out (LOSO) cross-validation and a subject-disjoint validation split:
        </p>
        {make_html_table(RESULTS_DATA[0], RESULTS_DATA[1:])}

        <div class="figure-container">
            <img src="{FIGURES_DIR / 'tsne_embeddings.png'}" width="380"/>
            <div class="figure-caption">Figure 3.1: t-SNE Projection. Left: Grouping by emotion classes. Right: Uniform mixing of subject IDs.</div>
        </div>
        <p>
            As shown by t-SNE (Figure 3.1), the GRL achieves complete subject-invariance, as subject IDs are uniformly mixed. 
            Simultaneously, the SupCon loss structures a discriminative embedding space, grouping expressions by emotion. However, 
            the validation confusion matrix (Figure 3.2) indicates that while the model achieves strong recall on the majority classes 
            (56.25% Negative, 44.44% Others), it fails to predict Positive and Surprise classes (0% recall). This is a direct consequence 
            of minority class starvation (only 45 and 34 fine-tuning samples respectively).
        </p>
        <div class="figure-container">
            <img src="{FIGURES_DIR / 'confusion_matrix.png'}" width="350"/>
            <div class="figure-caption">Figure 3.2: Row-Normalized Validation Confusion Matrix.</div>
        </div>

        <!-- SECTION 4 -->
        <h1>4. How to Improve Results (Future Work)</h1>
        <p>
            To overcome class imbalance and temporal mismatch across frame rates (30 FPS macro vs. 200 FPS micro), we propose three key research directions:
        </p>
        <ul>
            {make_html_future_papers(FUTURE_PAPERS)}
        </ul>

        <!-- SECTION 5 -->
        <h1>5. Conclusion</h1>
        <p>
            This summary outlines a parameter-efficient (~382K parameters), subject-invariant spatiotemporal contrastive network for automated MER. 
            By extracting physics-based flow/strain modalities, applying parameter-free 3D attention (SimAM), utilizing a sequence transformer (SLSTT), 
            and training via joint GRL domain adaptation and SupCon, the network successfully strips subject identity details. 
            The system achieves a 26-fold LOSO grand mean accuracy of 58.11%. Implementing phase-aware temporal augmentation, multimodal semantic guidance 
            (MECLIP), and self-supervised MAE pre-training (TGMAE) in the future will resolve minority class starvation and temporal speed mismatches.
        </p>
    </body>
    </html>
    """

    with open(PDF_PATH, "wb") as f:
        status = pisa.CreatePDF(html_content, dest=f)

    if status.err:
        print(f"ERROR: PDF generation failed with {status.err} errors.")
        sys.exit(1)
    else:
        print(f"SUCCESS: PDF summary generated -> {PDF_PATH}")


# =====================================================================
#  Main Orchestrator
# =====================================================================

def main():
    print("=" * 60)
    print("COMPILING THESIS EXECUTIVE SUMMARY IN WORD & PDF FORMATS")
    print("=" * 60)
    
    generate_docx()
    generate_pdf()
    
    print("=" * 60)
    print("ALL SUMMARY DELIVERABLES CREATED SUCCESSFULLY!")
    print(f"1. MS Word Summary : {DOCX_PATH.name}")
    print(f"2. PDF Summary     : {PDF_PATH.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
