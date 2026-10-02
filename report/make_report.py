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
    "Credit-card fraud is difficult to detect because positive cases are rare and patterns "
    "can differ across institutions. We use a local federated-learning simulation to study "
    "how clients trained on separate shards can contribute to a shared fraud model. "
    "Model updates and summary metrics still carry information, so our setup does not "
    "provide a formal privacy guarantee."
)

doc.add_paragraph(
    "We implemented a same-schema federated fraud-detection simulator with FedAvg, "
    "performance and robust baselines, and an experimental contribution-aware aggregator. "
    "The scorer combines local validation quality, update-scale reliability, hard-fraud "
    "utility, leave-one-out validation PR-AUC complementarity, and historical utility. "
    "Our current comparisons are mixed: contribution-aware scoring helps in some screens "
    "but is not consistently better than FedAvg or robust aggregation. We report the setup "
    "and its limitations alongside the results."
)

doc.add_paragraph(
    "We review related work in adaptive aggregation, trust-aware federated learning, and "
    "fraud detection, then describe our implemented same-schema training path, data "
    "preparation, baselines, and experiments. We also outline the heterogeneous encoder/torso "
    "prototype and the integration work that remains. Our results are exploratory because "
    "the number of seeds is small and the same held-out test set has been reused."
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
    ("6.2 Research Position and Open Questions", 2),
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
    "We study federated learning (FL) as a way for participants to train a shared model "
    "while keeping training shards separate in the learning loop. In our local simulator, "
    "clients train on their assigned CSV shards and send model updates and summary metrics "
    "to a central server. This setup models data locality; it does not implement secure "
    "aggregation, encryption, or differential privacy, and therefore does not provide a "
    "formal privacy guarantee [2]."
)

doc.add_paragraph(
    "Fraud patterns can differ across institutions, which motivates our study of "
    "collaborative model training under heterogeneous client data. In this local "
    "simulation, shards are kept separate during client training, but the server process "
    "can access reference and test files. We therefore treat data locality as a property "
    "of the experimental workflow, not as a privacy or security guarantee."
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
    "We study how a server should combine client updates when banks differ in data volume, "
    "fraud prevalence, and fraud patterns. FedAvg, single-signal weighting, and robust "
    "coordinate methods provide distinct baselines. Our experiments ask whether "
    "validation-based contribution signals improve on those methods in specific settings."
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
    "These characteristics motivate our evaluation of client weighting under "
    "heterogeneous fraud profiles. We test whether validation quality, update reliability, "
    "and marginal utility provide useful signals beyond sample-count weighting, while also "
    "checking whether those signals remain helpful under different partition and attack "
    "scenarios."
)

doc.add_heading("1.4 Need for the Proposed System", level=2)

doc.add_paragraph(
    "The studies covered in Chapter 2 explore different combinations of aggregation, "
    "client reliability, privacy, and non-IID learning. We use those approaches to frame "
    "our experiments on validation-based client utility and scorer ablations; our review "
    "does not attempt to represent every system in the field."
)

doc.add_paragraph(
    "We test whether validation quality, update reliability, and leave-one-out utility "
    "signals improve aggregation across non-IID scenarios, and whether the effect changes "
    "under attack or feature heterogeneity."
)

doc.add_heading("1.5 Aim of the Project", level=2)

doc.add_paragraph(
    "Our aim is to evaluate contribution-aware aggregation for credit-card fraud detection "
    "against FedAvg, performance-weighted, and robust baselines under non-IID client data. "
    "We use scorer ablations to measure whether the implemented signals add repeatable "
    "value."
)

doc.add_heading("1.6 Objectives", level=2)

objectives = [
    "We preprocess the credit-card data and build seeded non-IID client shards.",
    "We train a shared-schema MLP with FedAvg and alternative aggregation baselines.",
    "We implement and ablate validation quality, update reliability, hard-fraud utility, complementarity, and temporal utility signals.",
    "We compare methods using paired seeds, PR-AUC, ROC-AUC, and thresholded metrics.",
    "We document the implementation, experimental results, and remaining limitations.",
]
for i, obj in enumerate(objectives, 1):
    p = doc.add_paragraph(f"Objective {i}: {obj}")
    p.paragraph_format.left_indent = Cm(0.7)

doc.add_heading("1.7 Scope of the Project", level=2)

doc.add_paragraph(
    "Our current scope is a single-machine study using the public credit-card fraud "
    "dataset, a same-schema MLP, deterministic non-IID and stress-scenario partitions, "
    "and comparisons among FedAvg, weighted, robust, and contribution-aware aggregation. "
    "We have not implemented secure aggregation, differential privacy, or real-world "
    "multi-institution deployment. Heterogeneous encoder/torso training remains a "
    "prototype outside the current experiment loop."
)

doc.add_heading("1.8 Expected Outcomes", level=2)

doc.add_paragraph(
    "Our current outcomes are: (1) a working same-schema federated learning simulator; "
    "(2) a multi-dimensional client-scoring implementation; (3) paired comparisons against "
    "FedAvg and robust baselines across several scenarios; and (4) a documented set of "
    "limitations and follow-up experiments. Current results do not show consistent "
    "improvement over FedAvg."
)

add_figure(
    os.path.join(FIGURES_DIR, "fig1_system_architecture.png"),
    "Figure 1: High-level simulator architecture. Each client trains from a separate "
    "shard and returns an update plus validation metrics for server-side scoring and "
    "aggregation. The drawing does not imply a formal privacy mechanism.",
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
    "Among the sources we reviewed, approaches differ in how they weight clients, model "
    "reliability, and account for evolving contributions. Some use local accuracy or loss; "
    "others cluster clients by data similarity or focus on security. This comparison "
    "motivates our ablations of local quality, update reliability, leave-one-out utility, "
    "and temporal signals. It is limited to the papers cited here and is not an exhaustive "
    "survey of the field."
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
    "In the sources we reviewed, federated fraud systems address client weighting, "
    "reliability, privacy, and non-IID data in different ways. This motivates several "
    "questions for our experiments: whether leave-one-out utility signals improve "
    "aggregation; whether local quality, update reliability, and hard-fraud utility add "
    "distinct information; and whether client value changes across rounds. Our review is "
    "a starting point, not an exhaustive claim about all prior work."
)

doc.add_heading("2.7 Research Direction", level=2)

doc.add_paragraph(
    "We implemented a same-schema framework that scores each update using validation "
    "quality, update-scale reliability, hard-fraud utility, validation PR-AUC "
    "complementarity, and historical utility. Our current research direction is to test "
    "which signals help under specific client heterogeneity and attack scenarios, and how "
    "those results compare with existing methods. The heterogeneous encoder/torso path "
    "remains a separate prototype."
)

add_page_break()

# ==========================================================
# CHAPTER 3 — PROPOSED APPROACH
# ==========================================================

doc.add_heading("Chapter 3 — Proposed Approach", level=1)

doc.add_heading("3.1 Proposed System", level=2)

doc.add_paragraph(
    "Our implemented system simulates a central server and several clients, each trained "
    "from a separate shard of the processed transaction data. Clients return model updates "
    "and validation metrics to the server. The selected aggregation method updates the "
    "global model; the contribution-aware option derives client weights from the scorer. "
    "Because this is a local simulation without secure aggregation or differential privacy, "
    "we describe data locality rather than a formal privacy guarantee."
)

doc.add_heading("3.2 System Architecture", level=2)

doc.add_paragraph(
    "Figure 1 (shown in Chapter 1) summarizes our simulator. Each client trains on its "
    "assigned shard and returns model updates and validation metrics. The server can "
    "aggregate with FedAvg, robust methods, or contribution-aware weights, then "
    "redistributes the global state for the next configured round. This local setup does "
    "not add secure aggregation or differential privacy."
)

doc.add_heading("3.3 Overall Workflow", level=2)

add_figure(
    os.path.join(FIGURES_DIR, "fig2_training_workflow.png"),
    "Figure 2: Workflow of one simulated round. The server distributes the global "
    "model; clients train locally and return updates with validation metrics; the server "
    "applies the selected aggregation method.",
    width=6.2,
)

doc.add_paragraph(
    "Our runner executes a configured number of rounds. In each round, the server "
    "distributes the current global state, each client trains locally and returns an "
    "update with validation metrics, and the server applies the selected aggregation. "
    "It records validation metrics during training and, by default, evaluates the held-out "
    "test set once after the final round."
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
    "Our implemented scorer combines five signals for each client update. These definitions "
    "describe the current same-schema implementation; the heterogeneous design in "
    "`docs/heterogeneous_architecture.md` is not yet connected to the training loop."
)

doc.add_paragraph(
    "Quality. Local validation PR-AUC, shrunk toward the client cohort when its "
    "validation set contains few fraud examples.",
    style="List Bullet",
)
doc.add_paragraph(
    "Reliability (called trust in the code). A robust update-scale check with "
    "client-specific history. It is not identity verification or a detector for every attack.",
    style="List Bullet",
)
doc.add_paragraph(
    "Novelty utility. Leave-one-client-out log-probability gain on hard positive "
    "validation examples; parameter distance alone is not treated as usefulness.",
    style="List Bullet",
)
doc.add_paragraph(
    "Complementarity. Leave-one-client-out PR-AUC gain on the validation reference, "
    "with a paired, class-stratified bootstrap lower bound.",
    style="List Bullet",
)
doc.add_paragraph(
    "Temporal utility. An exponential moving average of historical marginal utility.",
    style="List Bullet",
)

add_figure(
    os.path.join(FIGURES_DIR, "fig3_scoring_pipeline.png"),
    "Figure 3: Multi-dimensional scoring pipeline. Each client's submitted weights are "
    "scored along five dimensions, normalized, and combined into an aggregation weight.",
    width=6.2,
)

doc.add_paragraph(
    "The implementation combines the five configured scores with fixed profile weights, "
    "then applies a temperature-scaled softmax across clients. These weights drive a "
    "weighted average of matching model states. The profiles are experimental ablations, "
    "not learned or proven-optimal weights."
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
    "We have implemented train/validation/test preprocessing, seeded scenario "
    "partitioning, a shared-schema MLP, local client training and validation, a server "
    "round loop, weighted and robust aggregators, an experimental contribution scorer, "
    "controlled update attacks, a centralized baseline, a results dashboard, and "
    "component tests. The heterogeneous encoder/torso and schema helpers are prototypes "
    "that are not yet integrated with the federated training loop."
)

doc.add_heading("5.2 Initial Implementation", level=2)

doc.add_paragraph(
    "Our experiment runner creates deterministic client shards per seed, reuses each "
    "seed's partition across the compared methods, and records per-round validation "
    "history plus final test metrics. The server can run FedAvg, loss/accuracy weighting, "
    "FedProx, coordinate median, trimmed mean, Krum, and contribution-aware aggregation. "
    "The final test set is evaluated once after training by default; the scorer and "
    "threshold calibration use a separate validation reference."
)

doc.add_heading("5.3 Preliminary Results", level=2)

doc.add_paragraph(
    "In the 20-round clean label-skew matrix, FedAvg averaged 0.7102 PR-AUC and "
    "contribution-aware aggregation averaged 0.7031 across three seeds; the paired "
    "difference was -0.0072 (95% seed-based interval [-0.0328, 0.0185]). In the "
    "10-round sign-flip screen, contribution-aware averaged 0.7141 PR-AUC versus "
    "FedAvg's 0.6840, but the paired interval crossed zero and median, trimmed mean, "
    "and Krum were slightly higher on mean PR-AUC. In the corrected feature-skew run, "
    "full scoring averaged 0.7170 versus 0.7119 for quality-only; the paired difference "
    "was +0.0051 with a wide interval [-0.0076, 0.0178]. These are exploratory results "
    "from a small number of seeds and a repeatedly used test set. We track all scenarios "
    "and caveats in `docs/phase2_status.md`."
)

doc.add_heading("5.4 Initial Observations", level=2)

doc.add_paragraph(
    "Because fraud is rare, we prioritize PR-AUC alongside ROC-AUC; thresholded metrics "
    "also depend on validation calibration. Our paired results vary by scenario: the "
    "contribution-aware method is not consistently ahead of FedAvg, and robust coordinate "
    "baselines lead in some screens. Under feature skew, the corrected full profile was "
    "nearly tied with quality-only, so the current evidence does not isolate a repeatable "
    "benefit from the extra utility dimensions."
)

doc.add_heading("5.5 Challenges Encountered", level=2)

doc.add_paragraph(
    "We have worked through several methodological challenges. Rare positive examples "
    "make client-level validation estimates noisy, so the quality score shrinks local "
    "PR-AUC toward the cohort when fraud counts are small. We also corrected an "
    "intermediate bootstrap implementation that sampled without replacement. Current "
    "screens use small seed counts and reuse a fixed test set, so seed intervals do not "
    "capture test-sample uncertainty or repeated-comparison bias. Calibrating update "
    "reliability without penalizing legitimate client heterogeneity remains open."
)

doc.add_heading("5.6 Current Limitations", level=2)

doc.add_paragraph(
    "Our evaluation remains limited by a small number of seeds, short scenario screens, "
    "and repeated use of the same held-out test set. The simulator runs on one machine "
    "and has no secure aggregation or differential privacy. The heterogeneous "
    "encoder/torso components are not integrated with `FedClient` or `FedServer`, and "
    "the current attack hooks do not cover adaptive targeted poisoning. These limits "
    "shape our next experiments."
)

add_page_break()

# ==========================================================
# CHAPTER 6 — FUTURE RESEARCH DIRECTION
# ==========================================================

doc.add_heading("Chapter 6 — Future Research Direction", level=1)

doc.add_heading("6.1 Possible Extensions", level=2)

doc.add_paragraph(
    "We see several possible extensions: graph-based client relations, hierarchical "
    "aggregation, and communication compression. Secure aggregation and differential "
    "privacy could also be investigated, with their threat models and guarantees evaluated "
    "explicitly. These mechanisms are not part of our current simulator."
)

doc.add_heading("6.2 Research Position and Open Questions", level=2)

doc.add_paragraph(
    "Our literature review considers related work in client valuation, update "
    "reliability, and fraud-focused federated learning. We are comparing those methods "
    "with our validation-based utility signals to understand whether the combination "
    "adds measurable value under feature heterogeneity. Current experiments are "
    "exploratory and do not yet establish a consistent advantage; the question remains "
    "open pending broader comparisons and stronger evaluation."
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
    "Our next experiments focus on longer paired runs with more seeds, a prespecified "
    "untouched evaluation set, and stronger targeted attacks. We also plan a true "
    "chronological train/validation/test split and additional ablations to evaluate the "
    "scorer under temporal change and feature heterogeneity. Privacy mechanisms such as "
    "noised scores or secure aggregation require a separate threat-model and utility study."
)

doc.add_heading("6.5 Expected Contributions", level=2)

doc.add_paragraph(
    "Our current work products include a multi-dimensional scorer, a reproducible "
    "experiment runner, scenario and attack hooks, a dashboard, and paired comparisons "
    "with established aggregation baselines. The evaluation remains exploratory: results "
    "vary by scenario, and we have not observed consistent improvement over FedAvg. We "
    "are using further ablations and literature comparisons to determine which scoring "
    "signals are useful and where the approach is applicable."
)

add_page_break()

# ==========================================================
# CONCLUSION
# ==========================================================

doc.add_heading("Conclusion", level=1)

doc.add_paragraph(
    "In this report, we have described our local federated fraud-detection simulator, "
    "the contribution-aware scorer, related work, and current experiments. Our literature "
    "review considers research on trust, security, clustering, and client valuation; it "
    "does not establish that multi-dimensional contribution scoring is new to the fraud "
    "domain."
)

doc.add_paragraph(
    "We have implemented data preprocessing, seeded scenario partitions, a same-schema "
    "federated training loop, several aggregation baselines, contribution scoring, and a "
    "results dashboard. Our evaluations show mixed performance across scenarios and do "
    "not demonstrate a consistent gain over FedAvg or robust baselines. The heterogeneous "
    "encoder/torso design remains a prototype, and the current test set has been reused "
    "across screens. We plan to expand paired seeds, use an untouched evaluation set, and "
    "continue comparing our scoring design with related work."
)

add_page_break()

# ==========================================================
# FUTURE SCOPE
# ==========================================================

doc.add_heading("Future Scope", level=1)

doc.add_paragraph(
    "We see several directions for extending the simulator: asynchronous client "
    "participation, client dropout, changing network conditions, and hierarchical or "
    "graph-based aggregation. Secure aggregation and differential privacy could also be "
    "investigated, with their threat models and guarantees evaluated explicitly. Further "
    "work could test other financial fraud datasets and domains after the current methods "
    "have been evaluated on a broader set of seeds and untouched test data."
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