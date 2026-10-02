"""
Generate the full project report as a DOCX file.
Structure follows the Project Review 1 recommended topics.
Figures are embedded between the relevant sections.
"""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "Project_Report.docx")

doc = Document()

# ==========================================================
# Styles
# ==========================================================

style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.paragraph_format.line_spacing = 1.15
style.paragraph_format.space_after = Pt(6)

for level, size, color in [
    ("Heading 1", 18, RGBColor(0x1F, 0x3A, 0x5F)),
    ("Heading 2", 14, RGBColor(0x2E, 0x5C, 0x8A)),
    ("Heading 3", 12, RGBColor(0x3A, 0x6E, 0xA5)),
]:
    h = doc.styles[level]
    h.font.name = "Calibri"
    h.font.size = Pt(size)
    h.font.color.rgb = color
    h.font.bold = True


def add_caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(10)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    return p


def add_figure(path, caption, width=6.0):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(caption)


def add_reference(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Cm(1)
    p.paragraph_format.first_line_indent = Cm(-1)
    return p


def add_page_break():
    doc.add_page_break()


# ==========================================================
# TITLE PAGE
# ==========================================================

for _ in range(4):
    doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Federated Learning for Credit Card Fraud Detection")
run.font.size = Pt(26)
run.font.bold = True
run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("with Multi-Dimensional Client Contribution Scoring")
run.font.size = Pt(16)
run.font.color.rgb = RGBColor(0x2E, 0x5C, 0x8A)

doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Project Report")
run.font.size = Pt(14)
run.font.italic = True

doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Project Review 1 — Recommended Topics")
run.font.size = Pt(12)
run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

add_page_break()

# ==========================================================
# ABSTRACT
# ==========================================================

doc.add_heading("Abstract", level=1)

doc.add_paragraph(
    "Credit card fraud remains one of the most persistent and financially damaging problems "
    "in the banking sector. In recent years, federated learning has emerged as a promising "
    "approach that allows multiple financial institutions to train a shared fraud detection "
    "model collaboratively without exchanging raw transaction data. This preserves customer "
    "privacy while still benefiting from the collective knowledge of all participants. "
    "However, in a real-world federation, banks are not equal: they differ in data quality, "
    "data distributions, fraud patterns, and even in their reliability as participants."
)

doc.add_paragraph(
    "This report presents a federated learning framework for credit card fraud detection in "
    "which each participating bank submits its locally trained model weights to a central "
    "server, and the server evaluates these submissions along multiple scoring dimensions — "
    "including quality, trust, novelty, complementarity, and temporal trend — before "
    "aggregating them into a global model. Instead of treating every bank uniformly (as in "
    "standard FedAvg) or weighting purely by dataset size, the proposed approach assigns "
    "each client a contribution score that reflects how valuable its update actually is to "
    "the federation. The scoring pipeline is designed to be privacy-preserving: the server "
    "never sees raw data, only model weights and computed metrics."
)

doc.add_paragraph(
    "The report begins with a background discussion of federated learning and fraud "
    "detection, followed by a review of recent related work in adaptive aggregation, "
    "trust-aware federated learning, and fraud-specific systems. The proposed system "
    "architecture and methodology are then described in detail, together with the data "
    "preparation and experimental plan. Preliminary work carried out so far — including "
    "data preprocessing, non-IID partitioning into four bank shards, and a centralized "
    "training baseline — is presented. Finally, future research directions, possible "
    "extensions, and the expected contributions of the work are discussed."
)

add_page_break()

# ==========================================================
# KEYWORDS
# ==========================================================

doc.add_heading("Keywords", level=1)

kw = [
    "Federated Learning",
    "Fraud Detection",
    "Client Contribution Scoring",
    "Trust-Aware Aggregation",
    "Non-IID Data",
    "Privacy Preservation",
    "Adaptive Weighting",
    "Credit Card Fraud",
]
doc.add_paragraph(", ".join(kw))

add_page_break()

# ==========================================================
# TABLE OF CONTENTS
# ==========================================================

doc.add_heading("Table of Contents", level=1)

toc_items = [
    ("Abstract", 1),
    ("Keywords", 1),
    ("List of Figures", 1),
    ("List of Tables", 1),
    ("List of Abbreviations", 1),
    ("Chapter 1 — Introduction", 1),
    ("1.1 Background", 2),
    ("1.2 Problem Statement", 2),
    ("1.3 Motivation", 2),
    ("1.4 Need for the Proposed System", 2),
    ("1.5 Aim of the Project", 2),
    ("1.6 Objectives", 2),
    ("1.7 Scope of the Project", 2),
    ("1.8 Expected Outcomes", 2),
    ("Chapter 2 — Literature Review", 1),
    ("2.1 Overview of Existing Research", 2),
    ("2.2 Review of Existing Methods", 2),
    ("2.3 Review of Related Systems / Approaches", 2),
    ("2.4 Comparison of Existing Approaches", 2),
    ("2.5 Limitations of Existing Work", 2),
    ("2.6 Identified Research Gap", 2),
    ("2.7 Research Direction", 2),
    ("Chapter 3 — Proposed Approach", 1),
    ("3.1 Proposed System", 2),
    ("3.2 System Architecture", 2),
    ("3.3 Overall Workflow", 2),
    ("3.4 Major Components / Modules", 2),
    ("3.5 Proposed Methodology", 2),
    ("3.6 Algorithms / Techniques Considered", 2),
    ("3.7 Alternative Approaches Considered", 2),
    ("3.8 Proposed Research Path", 2),
    ("Chapter 4 — Data & Experimental Plan", 1),
    ("4.1 Dataset / Data Sources", 2),
    ("4.2 Data Characteristics", 2),
    ("4.3 Data Preprocessing", 2),
    ("4.4 Data Distribution / Partitioning", 2),
    ("4.5 Experimental Setup", 2),
    ("4.6 Tools and Technologies", 2),
    ("4.7 Evaluation Metrics", 2),
    ("4.8 Baseline Methods", 2),
    ("4.9 Planned Experiments", 2),
    ("Chapter 5 — Preliminary Work / Current Progress", 1),
    ("5.1 Work Completed", 2),
    ("5.2 Initial Implementation", 2),
    ("5.3 Preliminary Results", 2),
    ("5.4 Initial Observations", 2),
    ("5.5 Challenges Encountered", 2),
    ("5.6 Current Limitations", 2),
    ("Chapter 6 — Future Research Direction", 1),
    ("6.1 Possible Extensions", 2),
    ("6.2 Potential Novelty", 2),
    ("6.3 Research Questions", 2),
    ("6.4 Future Experiments", 2),
    ("6.5 Expected Contributions", 2),
    ("Conclusion", 1),
    ("Future Scope", 1),
    ("References", 1),
    ("Appendices", 1),
]

for title, level in toc_items:
    p = doc.add_paragraph()
    run = p.add_run(title)
    if level == 1:
        run.font.bold = True
    p.paragraph_format.space_after = Pt(2)
    if level == 2:
        p.paragraph_format.left_indent = Cm(1)

add_page_break()

# ==========================================================
# LIST OF FIGURES
# ==========================================================

doc.add_heading("List of Figures", level=1)

figs = [
    "Figure 1: System Architecture",
    "Figure 2: Federated Learning Training Round Workflow",
    "Figure 3: Multi-Dimensional Scoring Pipeline",
    "Figure 4: Data Partitioning (Non-IID Split)",
    "Figure 5: Overall Methodology Flowchart",
    "Figure 6: FedAvg vs. Proposed Contribution-Aware Approach",
]
for f in figs:
    doc.add_paragraph(f, style="List Number")

add_page_break()

# ==========================================================
# LIST OF TABLES
# ==========================================================

doc.add_heading("List of Tables", level=1)

tables = [
    "Table 1: Comparison of Existing Approaches to Federated Fraud Detection",
    "Table 2: Data Characteristics of the Credit Card Fraud Dataset",
    "Table 3: Summary of Abbreviations",
]
for t in tables:
    doc.add_paragraph(t, style="List Number")

add_page_break()

# ==========================================================
# LIST OF ABBREVIATIONS
# ==========================================================

doc.add_heading("List of Abbreviations", level=1)

abbr = [
    ("FL", "Federated Learning"),
    ("FedAvg", "Federated Averaging"),
    ("FedProx", "Federated Proximal"),
    ("IID", "Independent and Identically Distributed"),
    ("Non-IID", "Non-Independent and Identically Distributed"),
    ("MLP", "Multi-Layer Perceptron"),
    ("AUC", "Area Under the Curve"),
    ("ROC", "Receiver Operating Characteristic"),
    ("F1", "Harmonic Mean of Precision and Recall"),
    ("SMOTE", "Synthetic Minority Over-sampling Technique"),
    ("DP", "Differential Privacy"),
    ("HE", "Homomorphic Encryption"),
    ("GDPR", "General Data Protection Regulation"),
    ("CCPA", "California Consumer Privacy Act"),
    ("SGD", "Stochastic Gradient Descent"),
    ("GNN", "Graph Neural Network"),
]

table = doc.add_table(rows=1, cols=2)
table.style = "Light Grid Accent 1"
hdr = table.rows[0].cells
hdr[0].text = "Abbreviation"
hdr[1].text = "Full Form"
for a, b in abbr:
    row = table.add_row().cells
    row[0].text = a
    row[1].text = b

add_page_break()

# ==========================================================
# CHAPTER 1 — INTRODUCTION
# ==========================================================

doc.add_heading("Chapter 1 — Introduction", level=1)

doc.add_heading("1.1 Background", level=2)

doc.add_paragraph(
    "The financial sector faces a constant and evolving threat from credit card fraud. "
    "Fraudsters continuously adapt their strategies, finding new ways to bypass detection "
    "systems. Traditional centralized approaches to fraud detection collect transaction data "
    "from multiple sources into a single repository and train a detection model on that "
    "combined data. While effective in principle, this approach raises serious concerns "
    "about data privacy, regulatory compliance, and the practical difficulty of sharing "
    "sensitive financial data between competing institutions [1]."
)

doc.add_paragraph(
    "Federated learning (FL) offers an alternative paradigm. In FL, multiple participants "
    "collaboratively train a shared model without ever sharing their raw data. Each "
    "participant trains a local model on its own private data and only exchanges model "
    "updates — typically weight vectors — with a central server. The server aggregates "
    "these updates to produce a global model, which is then redistributed for the next "
    "round of local training [2]. This approach respects data privacy while still allowing "
    "the federation to learn from diverse data sources."
)

doc.add_paragraph(
    "In the context of fraud detection, FL is particularly attractive because fraud "
    "patterns are highly localized: a bank operating in one region may encounter fraud "
    "schemes that another region has never seen. By pooling model knowledge rather than "
    "raw data, federated fraud detection can capture a broader picture of fraudulent "
    "behaviour while keeping each bank's transaction records private."
)

doc.add_heading("1.2 Problem Statement", level=2)

doc.add_paragraph(
    "Although federated learning provides a natural fit for cross-bank fraud detection, "
    "several challenges remain. The most significant is the issue of client heterogeneity. "
    "In any realistic federation, participating banks differ substantially in the amount "
    "of data they hold, the quality of that data, the types of fraud they encounter, and "
    "even their reliability as participants. A standard FedAvg aggregation treats all "
    "clients equally, which can lead to a global model that is biased towards institutions "
    "with more data, or worse, degraded by clients that submit noisy, irrelevant, or "
    "potentially malicious updates [3]."
)

doc.add_paragraph(
    "The problem addressed in this project is: how should a central server decide how much "
    "weight to give to each client's model update in a federated fraud detection system? "
    "Simple heuristics such as weighting by dataset size are inadequate because they do "
    "not account for the quality, trustworthiness, novelty, or complementarity of the "
    "knowledge that each client contributes. A more principled approach is needed — one "
    "that scores clients along multiple dimensions and uses those scores to guide "
    "aggregation."
)

doc.add_heading("1.3 Motivation", level=2)

doc.add_paragraph(
    "Fraud detection is uniquely challenging among machine learning tasks for several "
    "reasons. First, fraud is extremely rare — in the widely used Kaggle credit card "
    "dataset, only about 0.17% of transactions are fraudulent. Second, fraud is an "
    "adversarial problem: the fraudsters are actively trying to evade detection, so the "
    "statistical patterns are not static. Third, fraud data is highly sensitive, making it "
    "difficult or impossible to pool data across institutions."
)

doc.add_paragraph(
    "These characteristics make federated learning an appealing solution, but they also "
    "mean that a naive FL implementation may perform poorly. Banks with very different "
    "fraud profiles contribute very differently to the shared model. A bank that has "
    "recently experienced a novel fraud scheme has uniquely valuable knowledge that other "
    "banks would benefit from. Conversely, a bank whose update is largely redundant — "
    "because its fraud patterns are already well represented in the global model — "
    "contributes little new information. The motivation for this project is to build a "
    "federated system that can recognize these differences and act on them."
)

doc.add_heading("1.4 Need for the Proposed System", level=2)

doc.add_paragraph(
    "Existing federated learning frameworks for fraud detection, as surveyed in Chapter 2, "
    "generally fall into one of two camps. The first camp uses simple aggregation schemes "
    "such as FedAvg, possibly with weighting by local accuracy or loss. The second camp "
    "focuses on security — adding encryption, differential privacy, or malicious-client "
    "detection. Very few systems attempt to measure the actual contribution of each "
    "client's update in a multi-dimensional way, and fewer still track how contributions "
    "change over time."
)

doc.add_paragraph(
    "There is a clear need for a system that: (1) evaluates clients along multiple scoring "
    "dimensions simultaneously; (2) uses these scores to weight aggregation in a "
    "principled way; and (3) adapts dynamically as the fraud landscape and client "
    "behaviour evolve. This project aims to address that need."
)

doc.add_heading("1.5 Aim of the Project", level=2)

doc.add_paragraph(
    "The aim of this project is to design and implement a federated learning framework for "
    "credit card fraud detection in which client model updates are scored along multiple "
    "dimensions — quality, trust, novelty, complementarity, and temporal trend — and "
    "these scores are used to perform contribution-aware aggregation. The project also "
    "aims to evaluate the proposed framework against standard baselines such as FedAvg "
    "under non-IID data distributions."
)

doc.add_heading("1.6 Objectives", level=2)

objectives = [
    "To preprocess the credit card fraud dataset and partition it into non-IID client shards to simulate multiple banks.",
    "To implement a federated learning baseline (FedAvg) for fraud detection using an MLP model.",
    "To design a multi-dimensional client scoring pipeline covering quality, trust, novelty, complementarity, and temporal trend.",
    "To implement contribution-aware aggregation that uses the computed scores to weight client updates.",
    "To evaluate the proposed method against standard baselines using metrics such as ROC-AUC and F1-score.",
    "To document the system architecture, methodology, and results in a structured project report.",
]
for i, obj in enumerate(objectives, 1):
    p = doc.add_paragraph(f"Objective {i}: {obj}")
    p.paragraph_format.left_indent = Cm(0.7)

doc.add_heading("1.7 Scope of the Project", level=2)

doc.add_paragraph(
    "The scope of this project includes: the use of a publicly available credit card fraud "
    "dataset; simulation of four to eight client banks using Dirichlet-based non-IID "
    "partitioning; an MLP-based fraud detection model; a central server implementing "
    "FedAvg and contribution-aware aggregation; and an evaluation framework comparing "
    "these approaches. The project does not currently implement full cryptographic secure "
    "aggregation or differential privacy mechanisms, although these are discussed in "
    "future scope. Real-time deployment is outside the scope; all experiments are "
    "simulated locally."
)

doc.add_heading("1.8 Expected Outcomes", level=2)

doc.add_paragraph(
    "The expected outcomes of this project are: (1) a working federated learning baseline "
    "for fraud detection; (2) a multi-dimensional client scoring pipeline; (3) a "
    "contribution-aware aggregation method that outperforms FedAvg under non-IID "
    "conditions; (4) an experimental evaluation with results and analysis; and (5) a "
    "comprehensive project report documenting the entire process."
)

add_figure(
    os.path.join(FIGURES_DIR, "fig1_system_architecture.png"),
    "Figure 1: High-level system architecture. Each bank trains locally on private data; "
    "only model weights are uploaded to the central server, which scores them and "
    "aggregates a global model.",
    width=6.2,
)

add_page_break()

# ==========================================================
# CHAPTER 2 — LITERATURE REVIEW
# ==========================================================

doc.add_heading("Chapter 2 — Literature Review", level=1)

doc.add_heading("2.1 Overview of Existing Research", level=2)

doc.add_paragraph(
    "Federated learning has been applied to fraud detection and related financial "
    "problems in a growing body of recent research. This review focuses on works that "
    "combine federated learning with fraud detection, adaptive aggregation, client "
    "weighting, trust/reliability scoring, and communication/privacy enhancements. The "
    "four primary reference works reviewed in detail are: a scalable and trustworthy "
    "federated fraud detection framework (FedFraud) [1], a hierarchical privacy-preserving "
    "federated learning system (HiFraud) [4], an MSc project on self-adaptive federated "
    "learning for credit card fraud [5], and a federated learning framework with adaptive "
    "gradient clipping and encrypted aggregation (SecureFed+) [6]. A broader overview of "
    "relevant federated learning methods is also included [2], [3]."
)

doc.add_heading("2.2 Review of Existing Methods", level=2)

doc.add_paragraph(
    "Several aggregation and client-weighting strategies have been proposed in the "
    "literature. The earliest and most widely used is Federated Averaging (FedAvg) [2], "
    "in which the server averages client updates weighted by the number of local samples. "
    "FedAvg is simple and effective when client data is roughly IID, but degrades under "
    "significant data heterogeneity."
)

doc.add_paragraph(
    "To address heterogeneity, FedProx [3] introduces a proximal term that keeps local "
    "updates close to the global model, reducing drift caused by non-IID data. Other "
    "approaches adapt the aggregation weights based on local performance measures such as "
    "loss or accuracy. In the fraud domain, an adaptive FL framework proposed by Farooq "
    "et al. weights clients by their detection accuracy and applies resampling techniques "
    "such as SMOTE and Tomek links to handle class imbalance."
)

doc.add_heading("2.3 Review of Related Systems / Approaches", level=2)

doc.add_paragraph(
    "FedFraud [1] is a federated fraud detection framework that combines three key "
    "components: secure aggregation using pairwise masking and threshold sharing; "
    "trust-aware aggregation where each client receives a dynamic trust score based on "
    "the consistency and reliability of its updates; and asynchronous communication to "
    "accommodate clients with uneven connectivity. Models evaluated include federated "
    "random forest, 1D CNN, and LSTM, achieving an F1-score of 0.90 and an AUC of 0.96. "
    "The authors also report a reduction in gradient reconstruction success to 15% "
    "compared to 35% with standard FedAvg. While trust scoring is a key component of "
    "FedFraud, the trust score is based mainly on update consistency and reliability "
    "rather than on the novelty or complementary value of the knowledge contributed."
)

doc.add_paragraph(
    "HiFraud [4] is a hierarchical privacy-preserving federated learning system "
    "specifically designed for cross-institutional fraud detection. It addresses non-IID "
    "data distributions and extreme class imbalance through three innovations: "
    "fraud-aware dynamic clustering that groups institutions by fraud-pattern similarity "
    "while balancing rare-fraud representation; star-chain knowledge transfer that "
    "propagates detection knowledge across clusters with distillation to prevent "
    "catastrophic forgetting; and privacy-adaptive aggregation using Rényi differential "
    "privacy, where noise is calibrated based on the KL divergence between a client's "
    "distribution and the cluster average. Reported results include an AUC-ROC of 0.935 "
    "at a privacy budget of ε = 2.3, convergence in 30 rounds (compared to 49 for "
    "DP-FedAvg), and detection of novel fraud patterns within 3 hours inside clusters "
    "versus 24 hours for flat FL."
)

doc.add_paragraph(
    "SecureFed+ [6] is a federated learning framework for credit card fraud detection "
    "built on three co-designed components: adaptive gradient clipping with a threshold "
    "schedule τ_t = τ0 + αt that stabilizes convergence under non-IID data; gradient-level "
    "homomorphic encryption using the Paillier scheme (2048-bit) so that aggregation can "
    "be performed on encrypted gradients; and performance-weighted secure aggregation "
    "that down-weights malicious or low-quality updates. The framework achieves an "
    "accuracy of 92.53% and an AUC-ROC of 0.961, and the authors provide formal guarantees "
    "on convergence rate and Byzantine robustness."
)

doc.add_paragraph(
    "An MSc project on self-adaptive federated learning [5] focuses on clustering clients "
    "by computational capability and data characteristics using KMeans, then applying "
    "model pruning to accommodate resource-constrained clients. It evaluated the approach "
    "on the Kaggle credit card fraud dataset, reporting an accuracy of 90.39% with "
    "clustering and an F1-score of 90.01%. The author notes that quantization was "
    "designed but not separately evaluated, and acknowledges limitations including the "
    "small number of simulated clients (four) and the absence of client dropout handling."
)

doc.add_heading("2.4 Comparison of Existing Approaches", level=2)

doc.add_paragraph(
    "The table below summarizes the key characteristics of the reviewed approaches in "
    "terms of scoring dimensions used for aggregation, privacy mechanisms, and known "
    "limitations."
)

table = doc.add_table(rows=1, cols=5)
table.style = "Light Grid Accent 1"
table.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = table.rows[0].cells
hdr[0].text = "Approach"
hdr[1].text = "Scoring Dimensions"
hdr[2].text = "Privacy / Comm"
hdr[3].text = "Temporal Tracking"
hdr[4].text = "Key Limitation"

rows_data = [
    ("FedAvg [2]", "Sample count only", "None", "No", "Ignores client heterogeneity"),
    ("FedProx [3]", "Proximal term", "None", "No", "Tuned hyperparameter, not task-aware"),
    ("Farooq et al. [7]", "Local accuracy", "Not specified", "No", "Single-dimension weighting"),
    ("FedFraud [1]", "Trust (consistency)", "Masking, async", "Partial (reputation only)", "No knowledge novelty / complementarity"),
    ("HiFraud [4]", "Fraud-pattern similarity", "Renyi DP", "Partial (fraud periodicity)", "No per-update complementarity utility"),
    ("SecureFed+ [6]", "Performance quality", "Paillier HE", "Adaptive clipping schedule", "No cross-client contribution scoring"),
    ("MSC FL [5]", "Resource / data clustering", "Pruning, quantization", "No", "Quantization untested, few clients"),
]

for r in rows_data:
    row = table.add_row().cells
    for i, cell_text in enumerate(r):
        row[i].text = cell_text

add_caption("Table 1: Comparison of existing approaches to federated fraud detection.")

doc.add_heading("2.5 Limitations of Existing Work", level=2)

doc.add_paragraph(
    "A consistent pattern emerges from the reviewed literature. Most adaptive federated "
    "learning methods for fraud detection weight clients by a single local performance "
    "metric, such as accuracy or loss, or cluster clients by coarse data-similarity "
    "features. Very few explicitly measure the cross-client complementarity of knowledge "
    "— that is, how much one client's update improves the performance of other clients. "
    "HiFraud [4] clusters institutions by fraud-pattern similarity, but does not compute "
    "how a client's model improves others. No reviewed work formally tracks how a "
    "client's contribution utility evolves over time."
)

doc.add_paragraph(
    "From a privacy and communication perspective, works such as SecureFed+ [6] and "
    "FedFraud [1] provide strong security contributions, but these are largely decoupled "
    "from the question of which updates are actually valuable. Compression and "
    "communication optimization in the fraud-FL literature is limited; MSC FL [5] "
    "explores pruning and quantization, but the evaluation of these techniques is "
    "incomplete."
)

doc.add_heading("2.6 Identified Research Gap", level=2)

doc.add_paragraph(
    "Based on the review, the following gaps are identified. First, knowledge "
    "complementarity is not formalized in a way that is actionable during aggregation. "
    "Even when a system clusters clients by similarity, it does not quantify how one "
    "client's update improves another's performance. Second, aggregation weights are "
    "typically computed from a single metric. The opportunity to combine quality, trust, "
    "novelty, and complementarity into a multi-dimensional contribution score has not "
    "been thoroughly explored in the fraud domain. Third, the temporal dimension — "
    "tracking how a client's contribution value changes over rounds, identifying "
    "improving or deteriorating clients — is largely absent from the reviewed works. "
    "Fourth, communication and privacy decisions are not connected to contribution "
    "value: all updates are treated equally in terms of transmission, regardless of how "
    "valuable they are."
)

doc.add_heading("2.7 Research Direction", level=2)

doc.add_paragraph(
    "The research direction taken in this project is to design a federated learning "
    "framework in which each client's submitted model weights are evaluated along "
    "multiple dimensions — quality, trust, novelty, complementarity, and temporal trend "
    "— and these evaluations are combined into a contribution score that directly "
    "influences aggregation. This direction builds on the strengths of the reviewed "
    "works while addressing their most significant limitations. It is deliberately "
    "grounded in the fraud domain, where client heterogeneity and evolving fraud "
    "patterns make multi-dimensional contribution scoring particularly relevant."
)

add_page_break()

# ==========================================================
# CHAPTER 3 — PROPOSED APPROACH
# ==========================================================

doc.add_heading("Chapter 3 — Proposed Approach", level=1)

doc.add_heading("3.1 Proposed System", level=2)

doc.add_paragraph(
    "The proposed system is a contribution-aware federated learning framework for credit "
    "card fraud detection. The system comprises a central aggregation server and a set of "
    "participating banks (clients). Each bank owns a private, non-overlapping partition "
    "of transaction data and trains a local fraud detection model. Only model weights are "
    "shared with the central server. The server evaluates each client's submission along "
    "multiple scoring dimensions and computes a contribution score for that client. These "
    "scores are then used as weights in the aggregation step, producing a global model "
    "that reflects the collective, quality-weighted knowledge of the federation."
)

doc.add_heading("3.2 System Architecture", level=2)

doc.add_paragraph(
    "Figure 1 (shown in Chapter 1) illustrates the high-level system architecture. Each "
    "bank trains a local model on private transaction data. Model weights are uploaded to "
    "the central server, which contains two core modules: the multi-dimensional scoring "
    "module and the contribution-aware aggregation module. The scoring module computes "
    "per-client scores along five dimensions. The aggregation module combines the "
    "individual updates using the normalized contribution scores as weights. The "
    "resulting global model is distributed back to the banks for the next training round."
)

doc.add_heading("3.3 Overall Workflow", level=2)

add_figure(
    os.path.join(FIGURES_DIR, "fig2_training_workflow.png"),
    "Figure 2: Workflow of one federated training round. The server distributes the "
    "global model; clients train locally; weights are uploaded; the server scores each "
    "client and performs contribution-aware aggregation.",
    width=6.2,
)

doc.add_paragraph(
    "The overall workflow proceeds in rounds. In each round: (1) the server distributes "
    "the current global model to all clients; (2) each client trains the model on its "
    "local data for a number of local epochs; (3) the client uploads its updated model "
    "weights to the server; (4) the server scores each client's update along the five "
    "dimensions; (5) the server aggregates the updates using the computed contribution "
    "scores; and (6) the updated global model is distributed again. This loop continues "
    "for a fixed number of rounds or until convergence."
)

doc.add_heading("3.4 Major Components / Modules", level=2)

doc.add_paragraph(
    "The system consists of the following major modules. The data preparation module "
    "loads the fraud dataset, performs standardization and class handling, and splits the "
    "data into training and test sets. The data partitioning module applies Dirichlet "
    "partitioning to create non-IID shards that simulate different banks. The client "
    "training module trains an MLP-based fraud detection model on each bank's local "
    "shard. The scoring module computes multi-dimensional scores for each submitted "
    "update. The aggregation module combines updates using the contribution scores. "
    "Finally, the evaluation module computes global metrics on a held-out test set."
)

doc.add_heading("3.5 Proposed Methodology", level=2)

doc.add_paragraph(
    "The methodology is centered on the multi-dimensional scoring pipeline illustrated "
    "in Figure 3. For each client i, the server computes:"
)

doc.add_paragraph(
    "Quality score. The local performance of the client's model, estimated from "
    "validation loss or accuracy on a held-out portion of the client's own data, "
    "reported as a metric alongside the weights submission.",
    style="List Bullet",
)
doc.add_paragraph(
    "Trust score. A dynamic reputation measure based on the consistency of the client's "
    "updates across rounds, and the degree to which the client's submitted metrics align "
    "with what the server can verify indirectly. Trust is updated adaptively: consistent "
    "clients maintain or increase trust over time.",
    style="List Bullet",
)
doc.add_paragraph(
    "Novelty score. A measure of how much new information the client's update carries "
    "relative to the current global model, estimated by the deviation of the client's "
    "update from the global model direction.",
    style="List Bullet",
)
doc.add_paragraph(
    "Complementarity score. A measure of whether the client's update brings knowledge "
    "that other clients lack, estimated by evaluating the client's model on a synthetic "
    "or proxy validation mix representing other clients' distributions.",
    style="List Bullet",
)
doc.add_paragraph(
    "Temporal trend score. A measure of whether the client's contribution is improving, "
    "stable, or deteriorating over rounds, computed from the recent history of the other "
    "four scores via an exponential moving average or trend analysis.",
    style="List Bullet",
)

add_figure(
    os.path.join(FIGURES_DIR, "fig3_scoring_pipeline.png"),
    "Figure 3: Multi-dimensional scoring pipeline. Each client's submitted weights are "
    "scored along five dimensions, normalized, and combined into an aggregation weight.",
    width=6.2,
)

doc.add_paragraph(
    "Once the five scores are computed, they are normalized (for example by min-max or "
    "softmax scaling) and combined into a single contribution score via a weighted fusion "
    "or a small learned model. The final aggregation weight for each client is the "
    "softmax of the combined scores, ensuring the weights sum to one. The global model "
    "is then computed as a weighted average of client models."
)

doc.add_heading("3.6 Algorithms / Techniques Considered", level=2)

doc.add_paragraph(
    "The following algorithms and techniques were considered during the design: Federated "
    "Averaging (FedAvg) as the baseline; FedProx for handling heterogeneity; accuracy-"
    "weighted aggregation; loss-weighted aggregation; complementarity-weighted aggregation "
    "based on cross-impact scores; and a learned attention-based fusion of the five "
    "scoring dimensions."
)

doc.add_heading("3.7 Alternative Approaches Considered", level=2)

doc.add_paragraph(
    "Several alternative approaches were considered and set aside for the initial version "
    "of the system. A graph neural network (GNN) over client updates — constructing a "
    "similarity graph of client-update embeddings and learning aggregation weights — was "
    "considered but deferred due to complexity. Hierarchical or clustered aggregation, "
    "as in HiFraud [4], was considered but deferred in favour of a flat architecture for "
    "simplicity. Full homomorphic encryption of gradients, as in SecureFed+ [6], was "
    "deferred to future work because of computational cost. Selective reconstruction "
    "after compression was also considered as a future extension."
)

doc.add_heading("3.8 Proposed Research Path", level=2)

add_figure(
    os.path.join(FIGURES_DIR, "fig5_methodology.png"),
    "Figure 5: Overall methodology, from data preparation through non-IID partitioning, "
    "local training, scoring, contribution-aware aggregation, and iterative evaluation.",
    width=5.8,
)

doc.add_paragraph(
    "The research path proceeds incrementally. First, a working FedAvg baseline is "
    "established and evaluated. Second, multi-dimensional scoring is added, and the "
    "system compares contribution-aware aggregation against the baseline. Third, ablation "
    "studies isolate the effect of each dimension. Finally, additional experiments test "
    "robustness under varying numbers of clients, different degrees of non-IID data, and "
    "simulated changes in fraud patterns over time."
)

add_page_break()

# ==========================================================
# CHAPTER 4 — DATA & EXPERIMENTAL PLAN
# ==========================================================

doc.add_heading("Chapter 4 — Data & Experimental Plan", level=1)

doc.add_heading("4.1 Dataset / Data Sources", level=2)

doc.add_paragraph(
    "The primary dataset used in this project is the widely studied credit card fraud "
    "detection dataset (often referred to as the Kaggle credit card fraud dataset), "
    "originally collected and made available by Dal Pozzolo et al. The dataset contains "
    "transactions made by credit cards in September 2013 by European cardholders. It "
    "contains 284,807 transactions, of which 492 are fraudulent, representing an "
    "extreme class imbalance of approximately 0.172%."
)

doc.add_heading("4.2 Data Characteristics", level=2)

table = doc.add_table(rows=1, cols=2)
table.style = "Light Grid Accent 1"
table.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = table.rows[0].cells
hdr[0].text = "Characteristic"
hdr[1].text = "Value"

data_rows = [
    ("Number of transactions", "284,807"),
    ("Number of fraudulent transactions", "492 (0.172%)"),
    ("Number of features", "30 (V1–V28, Time, Amount)"),
    ("Target variable", "Class (0 = normal, 1 = fraud)"),
    ("Missing values", "None"),
    ("Duplicate rows", "Present (removed during preprocessing)"),
    ("Feature types", "28 PCA-transformed, Time, Amount"),
]
for r in data_rows:
    row = table.add_row().cells
    row[0].text = r[0]
    row[1].text = r[1]

add_caption("Table 2: Data characteristics of the credit card fraud dataset.")

doc.add_heading("4.3 Data Preprocessing", level=2)

doc.add_paragraph(
    "The preprocessing pipeline consists of the following steps: loading the raw "
    "transactions from CSV; inspecting the data for missing values and duplicates; "
    "removing duplicate rows; separating features and target; splitting the data into "
    "training and test sets (80/20) with stratification to preserve the fraud ratio; "
    "standardizing the Time and Amount features using a standard scaler fitted on the "
    "training set only; and saving the processed training and test sets. These steps "
    "are implemented in a Jupyter notebook (data/process.ipynb)."
)

doc.add_heading("4.4 Data Distribution / Partitioning", level=2)

add_figure(
    os.path.join(FIGURES_DIR, "fig4_data_partitioning.png"),
    "Figure 4: Data partitioning. The processed training set is split into non-IID "
    "client shards using a Dirichlet distribution, while the test set is kept aside for "
    "global evaluation.",
    width=6.2,
)

doc.add_paragraph(
    "To simulate a realistic federation of banks, the training set is partitioned into "
    "four non-IID shards using a Dirichlet distribution with concentration parameter "
    "α = 0.1. A smaller α produces stronger heterogeneity, meaning that different banks "
    "receive very different distributions of fraud patterns, which mirrors the real-world "
    "situation where different institutions see different fraud types. Each bank's shard "
    "is saved as a separate CSV file (bank_a.csv through bank_d.csv). This partitioning "
    "is implemented in partition/split_non_iid.py."
)

doc.add_heading("4.5 Experimental Setup", level=2)

doc.add_paragraph(
    "The experiments are conducted on the local machine. The federated learning loop is "
    "simulated with the four bank shards acting as clients. In each federated round, "
    "each client trains the global model on its local data for a small number of epochs "
    "(for example, 1–3 local epochs), then submits its weights. The server evaluates, "
    "scores, and aggregates. The full pipeline is evaluated on the held-out test set "
    "after each round."
)

doc.add_heading("4.6 Tools and Technologies", level=2)

tools = [
    "Python 3.14",
    "PyTorch (model definition, training, and inference)",
    "scikit-learn (evaluation metrics, preprocessing)",
    "pandas and NumPy (data handling)",
    "Matplotlib (visualization)",
    "python-docx (report generation)",
]
for t in tools:
    doc.add_paragraph(t, style="List Bullet")

doc.add_heading("4.7 Evaluation Metrics", level=2)

doc.add_paragraph(
    "The following metrics are used to evaluate the global model: ROC-AUC (area under the "
    "receiver operating characteristic curve), F1-score (harmonic mean of precision and "
    "recall), precision, recall, and validation loss. ROC-AUC is the primary metric "
    "because it is robust to the extreme class imbalance present in the data."
)

doc.add_heading("4.8 Baseline Methods", level=2)

doc.add_paragraph(
    "The proposed contribution-aware aggregation is compared against the following "
    "baselines: standard FedAvg (weighting by sample count); loss-weighted aggregation "
    "(weighting by inverse local validation loss); and accuracy-weighted aggregation "
    "(weighting by local validation accuracy). These baselines isolate the benefit of "
    "multi-dimensional scoring over simple single-metric weighting."
)

doc.add_heading("4.9 Planned Experiments", level=2)

experiments = [
    "Experiment 1: FedAvg baseline with four clients and α = 0.1; record global ROC-AUC and F1 over rounds.",
    "Experiment 2: Contribution-aware aggregation using quality and trust scores only; compare to FedAvg.",
    "Experiment 3: Full multi-dimensional scoring (quality, trust, novelty, complementarity, temporal); compare to FedAvg and single-metric baselines.",
    "Experiment 4: Ablation study — remove one dimension at a time and measure the effect on global performance.",
    "Experiment 5: Vary α (0.1, 0.5, 1.0) to test robustness under different heterogeneity levels.",
    "Experiment 6: Vary the number of clients (4, 8) to test scalability.",
    "Experiment 7: Simulate concept drift by swapping fraud patterns mid-training and observe whether temporal scoring helps the global model adapt.",
]
for i, e in enumerate(experiments, 1):
    doc.add_paragraph(e, style="List Number")

add_page_break()

# ==========================================================
# CHAPTER 5 — PRELIMINARY WORK / CURRENT PROGRESS
# ==========================================================

doc.add_heading("Chapter 5 — Preliminary Work / Current Progress", level=1)

doc.add_heading("5.1 Work Completed", level=2)

doc.add_paragraph(
    "The following components have been completed so far. Data preprocessing has been "
    "carried out in a Jupyter notebook, producing processed training and test CSV files. "
    "The non-IID partitioning script has been implemented, generating four bank shards "
    "with controlled heterogeneity using a Dirichlet distribution (α = 0.1). A fraud "
    "detection model class (MLP with two hidden layers of 64 and 32 units) has been "
    "implemented in PyTorch. A centralized training script has been implemented, "
    "training the model on the combined training set using binary cross-entropy with "
    "positive-class weighting to handle imbalance. A dataset loader class has been "
    "implemented to load the processed CSVs into PyTorch datasets and data loaders."
)

doc.add_heading("5.2 Initial Implementation", level=2)

doc.add_paragraph(
    "The initial implementation consists of the following files: dataset/fraud_dataset.py "
    "implements the FraudDataset class that reads a processed CSV and returns feature "
    "tensors and label tensors. models/fraud_model.py implements the FraudDetectionModel "
    "MLP. partition/split_non_iid.py implements the Dirichlet-based non-IID partitioning "
    "into four banks. training/train.py implements centralized training with validation "
    "and checkpointing of the best model. The centralized training serves as a "
    "performance upper-bound reference before the federated version is implemented."
)

doc.add_heading("5.3 Preliminary Results", level=2)

doc.add_paragraph(
    "The centralized training baseline was run for 20 epochs on the processed dataset "
    "with a batch size of 64 and a learning rate of 0.001. The model uses binary "
    "cross-entropy loss with positive-weighting proportional to the inverse fraud rate. "
    "Training progress and validation metrics (precision, recall, F1, ROC-AUC) are "
    "logged each epoch, and the best model checkpoint is saved. These metrics provide "
    "a reference point against which the federated approaches will be compared."
)

doc.add_heading("5.4 Initial Observations", level=2)

doc.add_paragraph(
    "Several observations have been made from the preliminary work. The extreme class "
    "imbalance (0.17% fraud) makes ROC-AUC the most informative metric, as accuracy is "
    "misleading when the vast majority of samples are negative. Positive-class weighting "
    "in the loss function is effective at compensating for the imbalance. The Dirichlet "
    "partitioning with α = 0.1 produces shards with noticeably different fraud "
    "concentrations, which is essential for meaningfully testing contribution-aware "
    "aggregation. The MLP architecture is sufficient as a baseline model for "
    "demonstrating the federated framework."
)

doc.add_heading("5.5 Challenges Encountered", level=2)

doc.add_paragraph(
    "Several challenges have been encountered. Balancing the class distribution within "
    "each client shard while maintaining non-IID characteristics requires careful "
    "partitioning design. Choosing sensible local training parameters (epochs, learning "
    "rate) for the federated loop that do not cause client drift requires experimentation. "
    "Designing scoring metrics that are meaningful yet computable from weights only "
    "(without raw data) is an ongoing challenge. The dataset path and environment setup "
    "also required attention to ensure reproducibility."
)

doc.add_heading("5.6 Current Limitations", level=2)

doc.add_paragraph(
    "The current implementation has notable limitations. The federated training loop is "
    "not yet fully implemented; only the centralized baseline exists. The scoring "
    "pipeline is designed but not yet implemented. Privacy mechanisms such as encrypted "
    "aggregation or differential privacy are not yet implemented. The simulated "
    "environment uses only four clients on a single machine, which does not capture "
    "real-world network conditions. These limitations define the immediate next steps "
    "of the project."
)

add_page_break()

# ==========================================================
# CHAPTER 6 — FUTURE RESEARCH DIRECTION
# ==========================================================

doc.add_heading("Chapter 6 — Future Research Direction", level=1)

doc.add_heading("6.1 Possible Extensions", level=2)

doc.add_paragraph(
    "Several extensions to the proposed system are possible. A graph-based client "
    "relation module could model similarity between clients and use a graph neural "
    "network to learn aggregation weights. Hierarchical aggregation could group clients "
    "into clusters and aggregate within and between clusters, similar to HiFraud [4]. "
    "Selective reconstruction after compression could couple communication efficiency "
    "with contribution value: high-value components of an update are transmitted with "
    "full fidelity while low-value components are compressed. Secure aggregation using "
    "homomorphic encryption or additive masking [1], [6] could be added to strengthen "
    "privacy guarantees. Differential privacy could be applied to the scoring metrics "
    "themselves."
)

doc.add_heading("6.2 Potential Novelty", level=2)

doc.add_paragraph(
    "The most significant potential novelty of this work is the systematic, "
    "multi-dimensional scoring of client contributions in a federated fraud detection "
    "setting, combined with the temporal tracking of these contributions over time. "
    "As noted in Chapter 2, the reviewed literature typically weights clients by a "
    "single metric or focuses on security mechanisms. Combining quality, trust, "
    "novelty, complementarity, and temporal trend into a single contribution score that "
    "directly drives aggregation is, to the best of our knowledge based on the reviewed "
    "literature, a direction that remains relatively unexplored in the fraud domain."
)

add_figure(
    os.path.join(FIGURES_DIR, "fig6_comparison.png"),
    "Figure 6: Comparison between standard FedAvg (sample-count weighting) and the "
    "proposed contribution-aware aggregation (multi-dimensional scoring).",
    width=6.2,
)

doc.add_heading("6.3 Research Questions", level=2)

doc.add_paragraph(
    "The following research questions guide the future direction of the project. "
    "Research Question 1: Does multi-dimensional contribution scoring improve global "
    "fraud detection performance compared to sample-count weighting under non-IID "
    "conditions? Research Question 2: Which scoring dimensions contribute most to the "
    "improvement, and how do they interact? Research Question 3: Does temporal tracking "
    "of contributions help the global model adapt when fraud patterns shift over time? "
    "Research Question 4: Can contribution scores be computed in a privacy-preserving "
    "manner without significantly reducing their effectiveness?"
)

doc.add_heading("6.4 Future Experiments", level=2)

doc.add_paragraph(
    "Future experiments will include: the full federated loop with contribution-aware "
    "aggregation; ablation studies for each scoring dimension; variation of the Dirichlet "
    "concentration parameter α; scaling to 8–16 clients; concept-drift simulation by "
    "modifying fraud patterns mid-training; comparison with FedProx and accuracy/loss-"
    "weighted baselines; and privacy-aware experiments with noised or masked updates."
)

doc.add_heading("6.5 Expected Contributions", level=2)

doc.add_paragraph(
    "The expected contributions of this project are: (1) a multi-dimensional client "
    "scoring framework for federated fraud detection; (2) an empirical evaluation "
    "demonstrating the value of contribution-aware aggregation under non-IID fraud data; "
    "(3) an analysis of which scoring dimensions matter most and how they interact; "
    "(4) a reproducible implementation consisting of data partitioning, training, "
    "scoring, and aggregation modules; and (5) a structured project report suitable "
    "for academic review."
)

add_page_break()

# ==========================================================
# CONCLUSION
# ==========================================================

doc.add_heading("Conclusion", level=1)

doc.add_paragraph(
    "This report has presented the motivation, background, related work, proposed "
    "approach, and preliminary progress for a federated learning framework for credit "
    "card fraud detection with multi-dimensional client contribution scoring. The "
    "review of recent literature — including FedFraud, HiFraud, SecureFed+, and an "
    "MSc project on self-adaptive FL — revealed that while trust, security, and "
    "clustering have received attention, a systematic multi-dimensional assessment of "
    "client contribution that is directly used in aggregation remains largely "
    "unaddressed in the fraud domain."
)

doc.add_paragraph(
    "The proposed system evaluates each bank's submitted model weights along five "
    "dimensions — quality, trust, novelty, complementarity, and temporal trend — and "
    "uses the combined scores to weight aggregation. The architecture, workflow, "
    "methodology, and experimental plan have been described. Preliminary work, "
    "including data preprocessing, non-IID partitioning, and a centralized baseline, "
    "has been completed and documented. The next steps are to implement the federated "
    "loop, the scoring pipeline, and the contribution-aware aggregator, and then to "
    "evaluate the system against the planned baselines."
)

add_page_break()

# ==========================================================
# FUTURE SCOPE
# ==========================================================

doc.add_heading("Future Scope", level=1)

doc.add_paragraph(
    "Beyond the immediate next steps, the project has considerable scope for "
    "extension. Real-world deployment considerations — such as asynchronous client "
    "participation, client dropout, and varying network conditions — could be "
    "simulated. Graph-based learning over client relationships could make aggregation "
    "more intelligent. Hierarchical federated learning could group banks by fraud "
    "profile. Secure aggregation and differential privacy could be integrated to "
    "provide formal privacy guarantees. Finally, the framework could be adapted to "
    "other financial fraud scenarios, such as money laundering detection or "
    "fraudulent loan applications, and to other sensitive domains such as healthcare."
)

add_page_break()

# ==========================================================
# REFERENCES (IEEE style)
# ==========================================================

doc.add_heading("References", level=1)

references = [
    "[1] Y. Alhasawi, A. A. Almtrf, and M. Asad, \u201cA federated approach to scalable and trustworthy financial fraud detection,\u201d unpublished manuscript.",
    "[2] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, \u201cCommunication-efficient learning of deep networks from decentralized data,\u201d in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), Fort Lauderdale, FL, USA, 2017, pp. 1273\u20131282.",
    "[3] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, \u201cFederated optimization in heterogeneous networks,\u201d in Proceedings of the 3rd MLSys Conference, Austin, TX, USA, 2020.",
    "[4] Z. Zhang, Z. Liu, X. Li, and L. Zhang, \u201cHiFraud: Hierarchical privacy-preserving federated learning with star-chain knowledge transfer for cross-institutional fraud detection,\u201d Computers, Materials & Continua, 2026.",
    "[5] MSc Student, \u201cSelf-adaptive federated learning for credit card fraud detection,\u201d M.S. thesis, National College of Ireland, 2024.",
    "[6] T. Manonmani, N. Umakanth, S. N. Kumar, S. Kannadhasan, S. K. Hareni, and P. Shree Suha Tharishana, \u201cSecureFed+: A federated learning framework with adaptive gradient clipping and encrypted aggregation for credit card fraud detection,\u201d International Journal of Machine Learning and Cybernetics, 2026.",
    "[7] Farooq et al., \u201cAI-driven adaptive federated learning for fraud detection,\u201d Wiley ACIS, 2025.",
]
for ref in references:
    add_reference(ref)

add_page_break()

# ==========================================================
# APPENDICES
# ==========================================================

doc.add_heading("Appendices", level=1)

doc.add_heading("Appendix A: Directory Structure", level=2)

structure = """project_root/
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── banks/
│   │       ├── bank_a.csv
│   │       ├── bank_b.csv
│   │       ├── bank_c.csv
│   │       └── bank_d.csv
├── dataset/
│   ├── __init__.py
│   └── fraud_dataset.py
├── models/
│   ├── __init__.py
│   └── fraud_model.py
├── partition/
│   ├── __init__.py
│   └── split_non_iid.py
├── training/
│   ├── train.py
│   ├── test_dataset.py
│   └── test_model.py
├── report/
│   ├── make_figures.py
│   └── figures/
│       ├── fig1_system_architecture.png
│       ├── fig2_training_workflow.png
│       ├── fig3_scoring_pipeline.png
│       ├── fig4_data_partitioning.png
│       ├── fig5_methodology.png
│       └── fig6_comparison.png
└── Executive Summary.pdf"""

for line in structure.split("\n"):
    doc.add_paragraph(line)

doc.add_heading("Appendix B: Key Equations", level=2)

doc.add_paragraph(
    "Federated Averaging: w_{t+1} = Σ (n_i / n) · w_i(t)"
)
doc.add_paragraph(
    "Dirichlet partitioning: p ~ Dir(α, α, ..., α)  (per class)"
)
doc.add_paragraph(
    "Contribution score combination: S_i = Σ_k λ_k · s_{i,k}   where k indexes the five dimensions"
)
doc.add_paragraph(
    "Aggregation weight: w_i = softmax(S_i) = exp(S_i) / Σ_j exp(S_j)"
)
doc.add_paragraph(
    "Exponential moving average for temporal trend: EMA_i(t) = β · EMA_i(t-1) + (1 - β) · s_i(t)"
)

# ==========================================================
# SAVE
# ==========================================================

doc.save(OUTPUT_PATH)
print(f"Report saved to: {OUTPUT_PATH}")