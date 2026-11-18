"""Build the paper in the course's "Project Template" format (Word).

The content is the paper in ``paper/main.tex``, reorganized into the template's
sections. The template file supplies the page setup, numbering and footer; this
script only replaces the body.

Usage
-----
    python paper_project_format/build_docx.py "<Project Template.docx>"
"""

from __future__ import annotations

import re
import shutil
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "paper" / "figures"
OUT = HERE / "Beyond_Fraud_Detection_Project.docx"

FONT = "Times New Roman"
BODY_SZ = 24          # half-points: 12 pt
SMALL_SZ = 18         # 9 pt for tables and captions
LINE = 276            # 1.15 line spacing
EMU_IN = 914400
TEXT_W_IN = 6.14      # page width minus the template's margins

# --------------------------------------------------------------------------- inline markup
# **bold**, *italic*


def runs(text: str, size: int = BODY_SZ, bold: bool = False) -> str:
    out = []
    for part in re.split(r"(\*\*.+?\*\*|\*.+?\*)", text):
        if not part:
            continue
        b, i, t = bold, False, part
        if part.startswith("**"):
            b, t = True, part[2:-2]
        elif part.startswith("*"):
            i, t = True, part[1:-1]
        rpr = (f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:cs="{FONT}"/>'
               + ("<w:b/><w:bCs/>" if b else "") + ("<w:i/><w:iCs/>" if i else "")
               + f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/><w:lang w:val="en-US"/>')
        out.append(f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(t)}</w:t></w:r>')
    return "".join(out)


def ppr(jc="both", after=100, before=0, keep_next=False, extra=""):
    return (f'<w:pPr>{extra}{"<w:keepNext/>" if keep_next else ""}'
            f'<w:spacing w:before="{before}" w:after="{after}" w:line="{LINE}" w:lineRule="auto"/>'
            f'<w:jc w:val="{jc}"/></w:pPr>')


def P(text, jc="both", after=40, size=BODY_SZ, bold=False, before=0, keep_next=False):
    return f"<w:p>{ppr(jc, after, before, keep_next)}{runs(text, size, bold)}</w:p>"


def H(text, level=0):
    """Numbered heading using the template's own list (I., II. / a., b.)."""
    num = f'<w:pStyle w:val="Prrafodelista"/><w:numPr><w:ilvl w:val="{level}"/><w:numId w:val="4"/></w:numPr>'
    return (f'<w:p><w:pPr>{num}<w:keepNext/><w:spacing w:before="{100 if level == 0 else 40}" w:after="40" '
            f'w:line="{LINE}" w:lineRule="auto"/><w:contextualSpacing w:val="0"/><w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:cs="{FONT}"/><w:b/><w:bCs/><w:sz w:val="{BODY_SZ}"/><w:szCs w:val="{BODY_SZ}"/></w:rPr></w:pPr>{runs(text, BODY_SZ, True)}</w:p>')


def LIST(items, numbered=False):
    """Bullets (template list 2) or decimal list (template list 3)."""
    out = []
    for k, it in enumerate(items, 1):
        if numbered:
            out.append(f'<w:p><w:pPr><w:spacing w:before="0" w:after="40" w:line="{LINE}" w:lineRule="auto"/>'
                       f'<w:ind w:left="360" w:hanging="360"/><w:jc w:val="both"/></w:pPr>{runs(f"{k}. " + it)}</w:p>')
        else:
            out.append(f'<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>'
                       f'<w:spacing w:before="0" w:after="40" w:line="{LINE}" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>'
                       f'{runs(it)}</w:p>')
    return "".join(out)


# --------------------------------------------------------------------------- tables
def TABLE(caption, header, rows, widths):
    total = 8838
    tw = [int(total * w) for w in widths]
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in tw)
    border = ('<w:tblBorders><w:top w:val="single" w:sz="8" w:color="000000"/>'
              '<w:bottom w:val="single" w:sz="8" w:color="000000"/>'
              '<w:insideH w:val="single" w:sz="4" w:color="BFBFBF"/></w:tblBorders>')

    def row(cells, head=False):
        tcs = []
        for w, c in zip(tw, cells):
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if head else ""
            tcs.append(f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shade}</w:tcPr>'
                       f'<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>'
                       f'{runs(c, SMALL_SZ, head)}</w:p></w:tc>')
        trpr = '<w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>' if head else '<w:trPr><w:cantSplit/></w:trPr>'
        return f"<w:tr>{trpr}{''.join(tcs)}</w:tr>"

    cap = P(caption, jc="left", after=40, size=SMALL_SZ, bold=False, before=60, keep_next=True)
    tbl = (f'<w:tbl><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/>{border}'
           f'<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="70" w:type="dxa"/><w:right w:w="70" w:type="dxa"/></w:tblCellMar>'
           f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{row(header, True)}{"".join(row(r) for r in rows)}</w:tbl>')
    return cap + tbl + P("", after=60)


# --------------------------------------------------------------------------- figures
IMAGES: list[tuple[str, Path]] = []   # (rId, path)


def png_size(path: Path):
    w, h = struct.unpack(">II", path.read_bytes()[16:24])
    return w, h


def drawing(path: Path, width_in: float, pid: int) -> str:
    rid = f"rIdImg{len(IMAGES) + 1}"
    IMAGES.append((rid, path))
    w, h = png_size(path)
    cx = int(width_in * EMU_IN)
    cy = int(cx * h / w)
    return (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{pid}" name="Figure {pid}"/>'
            f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="{pid}" name="{path.name}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
            f'</wp:inline></w:drawing></w:r>')


_pid = [100]


def FIGURE(panels, caption, width_in=2.95):
    """panels: list of (png name, sub-label) shown side by side in a borderless table."""
    cells = []
    cw = int(8838 / len(panels))
    for name, label in panels:
        _pid[0] += 1
        w_px, h_px = png_size(FIG / name)
        w_in = min(width_in, 1.45 * w_px / h_px)
        img = drawing(FIG / name, w_in, _pid[0])
        sub = P(label, jc="center", after=0, size=SMALL_SZ) if label else ""
        cells.append(f'<w:tc><w:tcPr><w:tcW w:w="{cw}" w:type="dxa"/><w:vAlign w:val="bottom"/></w:tcPr>'
                     f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="0" w:after="0"/><w:jc w:val="center"/></w:pPr>{img}</w:p>{sub}</w:tc>')
    tbl = (f'<w:tbl><w:tblPr><w:tblW w:w="8838" w:type="dxa"/><w:jc w:val="center"/>'
           f'<w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/><w:right w:val="nil"/>'
           f'<w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders><w:tblLayout w:type="fixed"/></w:tblPr>'
           f'<w:tblGrid>{"".join(f"<w:gridCol w:w=\"{cw}\"/>" for _ in panels)}</w:tblGrid>'
           f'<w:tr><w:trPr><w:cantSplit/></w:trPr>{"".join(cells)}</w:tr></w:tbl>')
    return tbl + P(caption, jc="both", after=100, size=SMALL_SZ, before=20)


# =========================================================================== CONTENT
B = []
add = B.append

TITLE = "Beyond Fraud Detection: Uncovering Anomalous Financial Behavior Through Unsupervised Learning"
add(P(f"**{TITLE}**", jc="center", after=100, size=28))
add(P("**Authors:** José Armando Melchor Soto, Rolando Fortanell Canedo, Emiliano Contreras Huerta, Ismael Téllez Antillón", jc="left", after=20))
add(P("**Emails:** jose.melchor@iteso.mx, rolando.fortanell@iteso.mx, emiliano.contreras@iteso.mx, ismael.tellez@iteso.mx", jc="left", after=20))
add(P("**Location and date:** Tlaquepaque, Jalisco, November 18, 2026", jc="left", after=120))

add(P("**Resumen**", jc="left", after=20))
add(P("La detección de fraude suele plantearse como un problema supervisado, pero casi nunca hay ejemplos etiquetados. Este trabajo busca transacciones que se comportan distinto al resto: se entrenó un Isolation Forest con 3,293 transacciones bancarias de Estados Unidos sin etiquetas, divididas por fecha, con un umbral que marca el 2% más inusual del entrenamiento, y se analizaron su evolución temporal, su estabilidad en diez semillas, sus características, SHAP y un mapa interactivo por estado. El ordenamiento fue estable (Spearman 0.90 frente a Jaccard 0.69), el umbral marcó 6.79% en los meses finales en vez de 2%, lo marcado destacó por su monto y el patrón geográfico dependió del tamaño de muestra. Un modelo solo con las 2,512 transacciones reales confirmó la deriva y el papel del monto; los patrones de noche, fin de semana y estado surgieron solo con filas sintéticas. Sin etiquetas, un modelo no supervisado ordena lo inusual de forma consistente y explicable si su umbral se actualiza."))
add(P("**Palabras clave:** detección de anomalías, aprendizaje no supervisado, detección de fraude, Isolation Forest, datos financieros", jc="left", after=100))
add(P("**Abstract**", jc="left", after=20))
add(P("Fraud detection is usually framed as a supervised problem, but labeled examples are rarely available. This work looks for transactions that behave differently: an Isolation Forest was trained on 3,293 unlabeled US bank transactions, split by date, with a threshold that flags the 2% most unusual training transactions, and its output was analyzed by time, stability across ten seeds, characteristics, SHAP, and an interactive state map. The ranking was stable (Spearman 0.90 against Jaccard 0.69), the threshold flagged 6.79% in the final months instead of 2%, flagged transactions stood out by amount, and the geographic pattern depended on sample size. A model trained only on the 2,512 real transactions confirmed the drift and the role of the amount; night, weekend, and state patterns appeared only with synthetic rows. Without labels, an unsupervised model ranks unusual transactions consistently and explainably if its threshold is updated."))
add(P("**Keywords:** anomaly detection, unsupervised learning, fraud detection, Isolation Forest, financial data", jc="left", after=80))

# ---------------------------------------------------------------- I. Introduction
add(H("Introduction"))
add(P("Finding the few unusual transactions among many is impractical by hand, and supervised detection needs labels that are often missing, while fraud changes over time. Anomaly detection instead finds observations that deviate from the general behavior (Chandola, Banerjee, and Kumar 2009), and Isolation Forest isolates those easier to separate, suiting anomalies with no known pattern (Liu, Ting, and Zhou 2008); yet a score alone does not show whether a deviation is meaningful, consistent, or how it is distributed. This work applies an Isolation Forest without labels and analyzes its flags (cases to investigate, not fraud) by time, stability, behavior, feature importance, and geography, characterizing unusual financial behavior and what can be concluded without labels."))

# ---------------------------------------------------------------- II. Objectives
add(H("Objectives"))
add(H("General Objective", 1))
add(P("To build a machine learning model capable of detecting possible fraud by analyzing transaction data and identifying unusual patterns in financial transactions, without relying on fraud labels."))
add(H("Specific Objectives", 1))
add(LIST([
    "**Identify atypical transactions:** implement an Isolation Forest model to detect transactions whose behavior deviates from the general pattern of the data, without requiring records previously labeled as fraudulent or legitimate.",
    "**Analyze behavioral patterns:** examine transaction characteristics such as amount, frequency, timing, and other available variables to identify anomalies that may be related to fraudulent activity.",
    "**Evaluate the model's performance:** assess the Isolation Forest results with anomaly detection criteria (stability across runs, behavior of the threshold over time, and feature importance) to determine its ability to identify suspicious transactions and to recognize its limitations in the absence of labeled data.",
], numbered=True))

# ---------------------------------------------------------------- III. Problem Statement
add(H("Problem Statement"))
add(P("The public *Bank Transaction Dataset for Fraud Detection* (2,512 US transactions, 495 accounts) has **no fraud label**, so precision, recall, or AUC cannot be computed and the model must be judged by its stability, interpretability, and sense. Anomalies are rare by design (2%), behavior changes over time so a past threshold may fail, and the data are small and cover only 27 of the 52 states and territories of the planned US map. The problem is to detect and characterize unusual transactions reliably and explainably, and to measure the limits of doing so, without a single confirmed fraud."))

# ---------------------------------------------------------------- IV. Hypothesis
add(H("Hypothesis"))
add(P("*Even without fraud labels, a model can find unusual transactions in a way that is consistent and that we can explain.* It is tested through five statements:"))
add(LIST([
    "**H1 (stability):** retrained with different seeds, the order of the most unusual transactions stays almost the same, even if the list of flagged ones changes a little.",
    "**H2 (time):** a threshold chosen with old data flags more than 2% of transactions in newer months, because behavior changes.",
    "**H3 (amount):** flagged transactions stand out mainly because of their amount, especially when it is large for its context.",
    "**H4 (many factors):** no single variable explains the anomaly score by itself.",
    "**H5 (geography):** states with few transactions show the most extreme rates, so the map cannot be read as real fraud risk.",
]))

# ---------------------------------------------------------------- V. Theoretical Framework
add(H("Theoretical Framework"))
add(P("**Anomaly detection.** An *anomaly* deviates clearly from most of the data (Chandola, Banerjee, and Kumar 2009). Isolation Forest isolates points with random cuts in many trees; anomalies, \"few and different,\" need fewer cuts and score higher (Liu, Ting, and Zhou 2008). The *anomaly score* lies in [0, 1], a transaction is *flagged* at the *threshold*, and the *contamination* (2%) is the expected flagged share. A flag means *unusual*, not fraud. No detector wins everywhere and global ones are a safe default (Goldstein and Uchida 2016); on labeled card data the best unsupervised model reached an AUC of 0.961 against 0.989 supervised (Niu, Wang, and Yang 2019). Axis-parallel cuts make scores variable, which random-slope cuts reduce (Hariri, Carrasco Kind, and Brunner 2021)."))
add(P("**Fraud detection and features.** Supervised methods face *class imbalance*, handled by SMOTE (Chawla et al. 2002), cost-sensitive thresholds (Danang and Putra 2025), or graph neural networks over accounts, merchants, and devices (Cheng et al. 2024), all needing labels. Customer-behavior summaries help (Whitrow et al. 2009), especially comparing a transaction with the customer's previous ones and usual hour (Correa Bahnsen et al. 2016)."))
add(P("**Drift, leakage, stability, and interpretation.** Habits and fraud strategies change (*concept drift*), so fixed models lose calibration (Gama et al. 2014; Dal Pozzolo et al. 2018), also seasonally (Lucas and Jurgovsky 2020). Mixing periods leaks future information (*data leakage*), avoided by a temporal split (Kaufman et al. 2012). Stability is how much a ranking or selected set changes under perturbed training (Kalousis, Prados, and Hilario 2007). SHAP shares the output among features via Shapley values (Lundberg and Lee 2017), efficiently for trees (Lundberg et al. 2020). Small groups have extreme rates (Wainer 2007), and synthetic data are only as faithful as their generator (Jordon et al. 2022)."))
add(P("**Key terms.** *Machine learning*: learning patterns from data; *deep learning*: many-layer neural networks; *supervised*, *semi-supervised*, *unsupervised*: with labels, few labels, none; *label*: the known answer; *feature (engineering)*: a variable (and creating better ones); *one-hot encoding*: a category as yes/no columns; *hyperparameter*: a preset setting; *random seed*: fixes random choices; *decision tree*: successive yes/no questions; *Jaccard*: set overlap (0-1); *Spearman*: agreement of orderings (-1 to 1); *standard deviation*: spread; *precision/recall*: share of flags that are fraud and of frauds flagged; *temporal split*: sets by date; *support*: transactions behind a statistic; *sanity check*: plausibility test, not proof; *choropleth*: regions colored by value; *GeoJSON*: region-shape file; *dashboard*: interactive web page; *synthetic data*: artificial data imitating real data."))

# ---------------------------------------------------------------- VI. Methodology
add(H("Methodology"))
add(P("Figure 1 summarizes the steps, replicable with the project repository (seed 42 unless stated)."))
add(FIGURE([("fig01_methodology.png", "(a) Methodology"), ("fig00_temporal_split.png", "(b) Temporal split")],
           "**Figure 1.** (a) Transactions are split by date, turned into 85 features, and scored by an Isolation Forest whose output is analyzed from five angles. (b) The model and threshold are fixed on training and applied unchanged to validation and test."))
add(LIST([
    "**Data.** Records hold amount, type, channel, date and time, city, customer age and occupation, session duration, login attempts, and balance. For the 25 states missing from the map, 781 identifiable *synthetic* rows were made by resampling real rows, assigning a city per state, and slightly perturbing amount, balance, and date (Figure 5a), giving 3,293 transactions from 654 accounts (2 January 2023 to 1 January 2024). Real transactions all occur on weekdays at 16:00-18:59; synthetic dates shift up to three days, covering every hour and day.",
    "**Features.** 85, computed on training data only: time (month, hour, night, weekend), amount (amount, balance, ratio), context (amount relative to the usual for the same hour, state, merchant, or account), account (transactions per account, login attempts), and one-hot state, type, channel, and occupation.",
    "**Temporal split.** Training 2 January-3 October 2023 (2,431), validation 4 October-2 November (302), test 3 November 2023-1 January 2024 (560), avoiding leakage (Kaufman et al. 2012).",
    "**Model and threshold.** Isolation Forest, 300 trees, scores in [0, 1]; threshold at the top 2% of training (0.710), a policy choice applied unchanged later; nothing was tuned after seeing results.",
    "**Stability, behavior, and geography.** Ten retrainings (seeds 42-51) scored on the same 560 test rows, compared pairwise by Jaccard of flagged sets and Spearman of rankings, counting runs per flagged row; flagged and normal test rows compared by difference of means in standard deviations; SHAP on the 560 test rows; a sanity check against a hand-made rule (high amount, many logins, unusual hour, long session); and the flagged share per state (minimum 15) shown in a web dashboard with an interactive US map (per-state transactions, flags, average score) and rates by hour, day, channel, and type.",
    "**Real-only comparison.** The unchanged pipeline re-run on the 2,512 real rows (*real-only* model, 60 features because the 25 synthetic-only state columns disappear) and compared with the *augmented* model on the same real test rows.",
], numbered=True))

# ---------------------------------------------------------------- VII. Results and Discussion
add(H("Results and Discussion"))
add(H("Detection, drift, and synthetic rows", 1))
add(P("The threshold flags 2.02% of training by design, 2.32% of validation, and 6.79% of test (Table 1), 3.4 times the target (4.6-6.8% over ten seeds). Overall, 108 transactions (3.28%) are flagged and hold 9.5% of the money. By month (Figure 2a) the rate stays below 2% until July, then reaches 3.2% (August), 5.4% (September), 6.8% (November), and 7.5% (December): gradual drift starting before the test period (Dal Pozzolo et al. 2018; Gama et al. 2014). Synthetic rows are flagged more (7.3% against 2.0%, Figure 2b), likely from rare perturbed combinations (Jordon et al. 2022), yet real rows also rise (1.5% to 4.2%): synthetic rows enlarge the drift, not cause it."))
add(TABLE("**Table 1.** Results per period (threshold = 0.710).",
          ["Period", "Transactions", "Flagged", "Rate", "Avg. score"],
          [["Train", "2,431", "49", "2.02%", "0.36"], ["Validation", "302", "7", "2.32%", "0.42"], ["Test", "560", "38", "6.79%", "0.42"]],
          [0.28, 0.18, 0.18, 0.18, 0.18]))
add(FIGURE([("fig05_monthly_drift.png", "(a) By month"), ("fig06_real_vs_synthetic.png", "(b) Real vs. synthetic rows")],
           "**Figure 2.** Share of flagged transactions (a) per month and (b) for real and synthetic rows per period; dashed line: 2% target."))
add(H("Stability, behavior, timing, and feature importance", 1))
add(P("Runs agree more on the *order* than on *which* transactions are flagged (Figure 3a): Spearman 0.90 against Jaccard 0.69. Of 560 test transactions, 508 are never flagged; of the other 52, 21 are borderline (one to four runs) and **16 are flagged in all ten runs**, all with very high scores. Disagreement sits near the threshold, partly from random axis-parallel cuts (Hariri, Carrasco Kind, and Brunner 2021). Flagged transactions are large (Figure 3b), $878 against $273, and large *for their context*: the top five differences (1.8-1.9 standard deviations) are amount features, then weekend timing (34% of flagged against 4%), students, night, and slightly more login attempts."))
add(FIGURE([("fig07_stability.png", "(a) Stability over ten seeds"), ("fig08_feature_contrast.png", "(b) Flagged vs. normal")],
           "**Figure 3.** (a) Agreement between runs and number of runs in which each flagged test transaction was flagged. (b) Largest flagged-normal differences in standard deviations; dark bars are amount-related."))
add(P("Transactions at 00:00-05:00 are flagged 19% of the time against 2.4%, and weekend ones 17-19% against 1.5-3.2% (Figure 4a); both are rare (about 5% and 7%), rarity is easy to isolate (Liu, Ting, and Zhou 2008), and no real transaction occurs then, so the pattern is synthetic. SHAP (Figure 4b) ranks the month, Texas, debit type, amount relative to the account, and California first, then occupation and other states; five feature types share the influence (maximum 0.147). SHAP explains *all* scores (common traits lower most; the month leads as test months follow training), not just the flagged few. The score agrees moderately with the hand-made rule (Spearman 0.47) but shares only 9% of top cases: plausible, not a copy, not proof."))
add(FIGURE([("fig10_temporal.png", "(a) By hour and day"), ("fig09_shap.png", "(b) SHAP importance")],
           "**Figure 4.** (a) Flagged share by hour and day; dark bars mark early hours and weekends. (b) Top 15 SHAP features by type."))
add(FIGURE([("fig02_dataset_map.png", "(a) Real and synthetic states"), ("fig11_state_map.png", "(b) Flagged rate by state")],
           "**Figure 5.** (a) States with real (dark) and only synthetic (light) transactions. (b) Flagged share per state, static version of the interactive map."))
add(H("Geography", 1))
add(P("The highest rates (Figure 5b) are Connecticut (17%), Alaska and West Virginia (15%), Kansas (14%), and Minnesota (11%); Texas and California are near 3%. This is not regional risk: Connecticut's 17% is 6 of 36 transactions, states with 40 or fewer transactions (20) are flagged at 7.9% against 2.2% for the 7 with 100 or more, and rate falls with size (Spearman -0.39, p = 0.004), the small-sample effect of Wainer (2007). All twelve highest-rate states contain only synthetic rows."))
add(H("Real-only versus augmented model", 1))
add(P("**Unchanged with real data** (Table 2): the drift persists (ten-seed test rates 4.0-5.6%; below 1% in February, over 2% in June, 5.4% in December, earlier than in the augmented model, Figure 6a), whose synthetic rows raise its threshold and lower its level on the same rows (H2). On the 427 real test rows both scores correlate (Spearman 0.82) and share 15 of the top 20, but flags agree less (22 and 19, 14 shared, Jaccard 0.52), and the real-only model is at least as stable (H1). The top five real differences are amount-related (1.9-2.7 standard deviations; H3), and SHAP spreads over geographic (42%), account, channel and type (27%), amount (17%), demographic (8%), and temporal (6%) features (H4). **Synthetic-dependent:** all 188 night and 214 weekend transactions are synthetic (24% of rows, 57 of 108 flags); the month's SHAP lead is not robust; and the 27 real states (41-357 transactions, rates 0%-7.3%, top Nevada with 55) show no significant rate-size relation (0.09, p = 0.66 for the augmented model on them), which appears only with synthetic-only states, including all 20 with 40 or fewer transactions and the twelve highest-rate ones (Figure 6b). Behavioral conclusions should rest on real data, without over-reading 22 flagged real test rows."))
add(TABLE("**Table 2.** Real-only versus augmented model.",
          ["Measure", "Real-only", "Augmented"],
          [["Rows (train / val. / test)", "1,867 / 218 / 427", "2,431 / 302 / 560"],
           ["Features; threshold", "60; 0.685", "85; 0.710"],
           ["Flagged %, own periods", "2.04 / 3.67 / 5.15", "2.02 / 2.32 / 6.79"],
           ["Flagged %, real rows only", "2.09 / 4.13 / 4.68", "1.50 / 2.29 / 4.22"],
           ["Spearman / Jaccard across seeds", "0.92 / 0.75", "0.90 / 0.69 (real rows: 0.88 / 0.63)"],
           ["Flagged in all ten runs", "12 of 427", "16 of 560 (real rows: 5 of 427)"],
           ["Test amount, flagged vs. normal", "$1,080 vs. $262", "$878 vs. $273"],
           ["Flagged at night or weekend", "0", "48 of 108"],
           ["Rank of month in SHAP", "13 of 60", "1 of 85"],
           ["States; rate vs. size (Spearman)", "27; -0.30 (p = 0.13)", "52; -0.39 (p = 0.004)"]],
          [0.40, 0.25, 0.35]))
add(FIGURE([("fig13_real_vs_augmented.png", "(a) Real rows, both models"), ("fig14_state_support_comparison.png", "(b) Rate vs. state size")],
           "**Figure 6.** (a) Flagged share of real transactions under both models, by period and month. (b) Flagged rate against transactions per state; extreme rates belong to small synthetic-only states."))
add(H("Comparison with previous studies", 1))
add(P("Cited studies describe fraud and this model unusual transactions, so agreement shows plausibility only (Table 3). **Night:** Afriyie et al. (2023) used 555,719 labeled *simulated* western-US transactions; Accertify (2026), 576 million retail transactions in the 2021-2025 Cyber Weeks, reports overnight fraud at 2.1 times midday; Stripe saw late-night peaks (Woodward 2017). Here the 22-04 h rate (15.0%) is about 5 times midday (2.8%, 11:00-13:59), stronger because night is rarer (79% of transactions at 16:00-18:59) and synthetic. **Weekends:** Accertify's 2025 fraud-to-legitimate cart value ratio was 2.56 on weekends and Afriyie et al. counted 372 Sunday frauds (a count); here weekends reach 17.8% against 2.3%, read with caution. **Amount:** fraud amounts have a higher median in Afriyie et al. **Profile:** victims differ from unusual transactions; flagged accounts have lower balances, retired customers are average (2.4%), students reach 5.9% (others 2.3-2.4%). **Time:** Dal Pozzolo et al. (2018) saw drift over 75 million real transactions in three years; here it also appears on real rows and calls for periodic updates. Without labels, accuracy is not measurable."))
add(TABLE("**Table 3.** Results compared with previous studies.",
          ["Topic", "Previous studies", "This work", "Agree?"],
          [["Night hours", "22-04 h: 12.7% of transactions, 19.7% of fraud (Accertify 2026); fraud at 22-04 h (Afriyie et al. 2023)", "22-04 h: 6.5% of transactions, 29.6% of flagged", "Yes"],
           ["Weekends", "Fraud rises on quiet days (Woodward 2017); most frauds on Sunday (Afriyie et al. 2023)", "6.5% of transactions, 35.2% of flagged", "Partly"],
           ["Amount", "Customer-relative features help (Whitrow et al. 2009; Correa Bahnsen et al. 2016)", "Top 5 differences are amount vs. context", "Yes"],
           ["Profile", "Victims over 60 most frequent (Afriyie et al. 2023)", "Students flagged most (5.9%)", "No"],
           ["Time", "Concept drift in fraud (Dal Pozzolo et al. 2018; Gama et al. 2014)", "2.0% to 6.8% flagged", "Yes"],
           ["Stability", "IF scores vary with cuts (Hariri, Carrasco Kind, and Brunner 2021); stability measures (Kalousis, Prados, and Hilario 2007)", "Ranking 0.90 vs. set 0.69", "Yes"],
           ["Geography", "Small areas show extreme rates (Wainer 2007)", "Small states: 7.9% vs. 2.2%", "Yes"],
           ["Method", "Unsupervised AUC 0.961 vs. 0.989 (Niu, Wang, and Yang 2019); global detectors (Goldstein and Uchida 2016)", "No labels to measure AUC", "-"]],
          [0.13, 0.44, 0.33, 0.10]))
add(H("Hypothesis evaluation and limitations", 1))
add(P("Five of eight compared results agree with previous studies, one partly, one differs, and one cannot be checked. All hypotheses are supported (Table 4), H5 more than expected given the synthetic rows; the real-only model supports H1-H4 (H5 is untestable there, every real state having 41 or more transactions). Reviews should follow the *score*, the most useful output, with the threshold as a periodically updated budget; without labels, drift cannot be told apart from behavior change, seasonality (one year), or noise. **Limitations:** flags are not fraud and rare legitimate behavior is flagged; a fixed threshold fails over time; 781 rows (a quarter) are synthetic and the only night and weekend data; states are small and the test period has 560 rows; results are descriptive; validation needs labels, analyst feedback, or another dataset."))
add(TABLE("**Table 4.** Hypothesis evaluation.",
          ["H", "Main evidence", "Result"],
          [["H1", "Ranking 0.90 vs. flag 0.69; 16 always flagged; also in real-only model", "Supported"],
           ["H2", "2.0% to 2.3% to 6.8%; also with real rows only", "Supported"],
           ["H3", "Top five differences are all amount-related", "Supported"],
           ["H4", "Five feature types in the SHAP top 15; maximum 0.147", "Supported"],
           ["H5", "Small and synthetic states have the highest rates", "Supported"]],
          [0.08, 0.72, 0.20]))

# ---------------------------------------------------------------- VIII. Conclusions
add(H("Conclusions"))
add(P("Without labels, the Isolation Forest ranked transactions consistently across ten runs (16 always flagged); its past threshold flagged three times more in the final months as scores rose from August, also in real data; flags stand out by amount while the model uses many factors; and the map reflects sample size and synthetic rows, the sole source of night, weekend, and state patterns. This agrees with prior studies. **The hypothesis is proven** (H1-H5; H1-H4 also with real data only). **The general objective was met**, as were the specific ones: (1) atypical transactions were identified with an Isolation Forest; (2) amount, timing, frequency, and other variables were analyzed, with amount relative to context as the main signal; (3) stability, threshold behavior, and SHAP evaluated performance and exposed its limits. The impact is a reproducible, explainable way to prioritize reviews without labels; future work includes periodic threshold updates, multi-year real data with nights and weekends, other detectors, and analyst-reviewed stable anomalies as first labels for semi-supervised learning."))

# ---------------------------------------------------------------- IX. Bibliography (Chicago, author-date)
add(H("Bibliography"))
BIB = [
    "Accertify. 2026. \"Cyber Week Fraud Trends: 5 Seasons of Retail Data.\" September 8, 2026. https://www.accertify.com/resource/5-seasons-retail-data-shows-how-cyber-week-has-changed/.",
    "Afriyie, Jonathan Kwaku, Kassim Tawiah, Wilhemina Adoma Pels, Sandra Addai-Henne, Harriet Achiaa Dwamena, Emmanuel Odame Owiredu, Samuel Amening Ayeh, and John Eshun. 2023. \"A Supervised Machine Learning Algorithm for Detecting and Predicting Fraud in Credit Card Transactions.\" *Decision Analytics Journal* 6: 100163. https://doi.org/10.1016/j.dajour.2023.100163.",
    "Chandola, Varun, Arindam Banerjee, and Vipin Kumar. 2009. \"Anomaly Detection: A Survey.\" *ACM Computing Surveys* 41 (3): 1-58.",
    "Chawla, Nitesh V., Kevin W. Bowyer, Lawrence O. Hall, and W. Philip Kegelmeyer. 2002. \"SMOTE: Synthetic Minority Over-sampling Technique.\" *Journal of Artificial Intelligence Research* 16: 321-57.",
    "Cheng, Dawei, Yao Zou, Sheng Xiang, and Changjun Jiang. 2024. \"Graph Neural Networks for Financial Fraud Detection: A Review.\" *Frontiers of Computer Science*.",
    "Correa Bahnsen, Alejandro, Djamila Aouada, Aleksandar Stojanovic, and Björn Ottersten. 2016. \"Feature Engineering Strategies for Credit Card Fraud Detection.\" *Expert Systems with Applications* 51: 134-42. https://doi.org/10.1016/j.eswa.2015.12.030.",
    "Dal Pozzolo, Andrea, Giacomo Boracchi, Olivier Caelen, Cesare Alippi, and Gianluca Bontempi. 2018. \"Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy.\" *IEEE Transactions on Neural Networks and Learning Systems* 29 (8): 3784-97.",
    "Danang, and T. W. A. Putra. 2025. \"Cost-Sensitive Fraud Detection with Reliability Calibration: XGBoost and Focal-Proxy Reweighting.\" *International Journal of Information Technology and Business*.",
    "Gama, João, Indrė Žliobaitė, Albert Bifet, Mykola Pechenizkiy, and Abdelhamid Bouchachia. 2014. \"A Survey on Concept Drift Adaptation.\" *ACM Computing Surveys* 46 (4): 44:1-44:37. https://doi.org/10.1145/2523813.",
    "Goldstein, Markus, and Seiichi Uchida. 2016. \"A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms for Multivariate Data.\" *PLoS ONE* 11 (4): e0152173. https://doi.org/10.1371/journal.pone.0152173.",
    "Hariri, Sahand, Matias Carrasco Kind, and Robert J. Brunner. 2021. \"Extended Isolation Forest.\" *IEEE Transactions on Knowledge and Data Engineering* 33 (4): 1479-89. https://doi.org/10.1109/TKDE.2019.2947676.",
    "Jordon, James, Lukasz Szpruch, Florimond Houssiau, Mirko Bottarelli, Giovanni Cherubin, Carsten Maple, Samuel N. Cohen, and Adrian Weller. 2022. \"Synthetic Data - What, Why and How?\" arXiv:2205.03257.",
    "Kalousis, Alexandros, Julien Prados, and Mélanie Hilario. 2007. \"Stability of Feature Selection Algorithms: A Study on High-Dimensional Spaces.\" *Knowledge and Information Systems* 12 (1): 95-116. https://doi.org/10.1007/s10115-006-0040-8.",
    "Kaufman, Shachar, Saharon Rosset, Claudia Perlich, and Ori Stitelman. 2012. \"Leakage in Data Mining: Formulation, Detection, and Avoidance.\" *ACM Transactions on Knowledge Discovery from Data* 6 (4): 15:1-15:21. https://doi.org/10.1145/2382577.2382579.",
    "Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. 2008. \"Isolation Forest.\" In *Proceedings of the IEEE International Conference on Data Mining (ICDM)*, 413-22.",
    "Lucas, Yvan, and Johannes Jurgovsky. 2020. \"Credit Card Fraud Detection Using Machine Learning: A Survey.\" arXiv:2010.06479.",
    "Lundberg, Scott M., Gabriel Erion, Hugh Chen, Alex DeGrave, Jordan M. Prutkin, Bala Nair, Ronit Katz, Jonathan Himmelfarb, Nisha Bansal, and Su-In Lee. 2020. \"From Local Explanations to Global Understanding with Explainable AI for Trees.\" *Nature Machine Intelligence* 2 (1): 56-67.",
    "Lundberg, Scott M., and Su-In Lee. 2017. \"A Unified Approach to Interpreting Model Predictions.\" In *Advances in Neural Information Processing Systems 30*, 4765-74.",
    "Niu, Xuetong, Li Wang, and Xulei Yang. 2019. \"A Comparison Study of Credit Card Fraud Detection: Supervised versus Unsupervised.\" arXiv:1904.10604.",
    "Wainer, Howard. 2007. \"The Most Dangerous Equation.\" *American Scientist* 95 (3): 249-56.",
    "Whitrow, Christopher, David J. Hand, Piotr Juszczak, David J. Weston, and Niall M. Adams. 2009. \"Transaction Aggregation as a Strategy for Credit Card Fraud Detection.\" *Data Mining and Knowledge Discovery* 18 (1): 30-55. https://doi.org/10.1007/s10618-008-0116-z.",
    "Woodward, Kevin. 2017. \"Quiet Online Shopping Days Yield the Biggest Fraud Spikes, Stripe Says.\" *Digital Transactions*, December 14, 2017. https://www.digitaltransactions.net/?p=21420.",
]
for ref in BIB:
    add(f'<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="{LINE}" w:lineRule="auto"/>'
        f'<w:ind w:left="720" w:hanging="720"/><w:jc w:val="left"/></w:pPr>{runs(ref)}</w:p>')


# =========================================================================== ASSEMBLY
def main(template: Path) -> None:
    work = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(template) as z:
        z.extractall(work)

    doc = work / "word" / "document.xml"
    x = doc.read_text(encoding="utf8")
    head = x[: x.index("<w:body>") + len("<w:body>")]
    sect = re.search(r"<w:sectPr.*?</w:sectPr>", x, flags=re.S).group(0)
    if "xmlns:wp=" not in head:
        raise SystemExit("template is missing the drawing namespace")
    doc.write_text(head + "".join(B) + sect + "</w:body></w:document>", encoding="utf8")

    media = work / "word" / "media"
    media.mkdir(exist_ok=True)
    rels_path = work / "word" / "_rels" / "document.xml.rels"
    rels = rels_path.read_text(encoding="utf8")
    new = []
    for rid, path in IMAGES:
        shutil.copy(path, media / path.name)
        new.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{path.name}"/>')
    rels_path.write_text(rels.replace("</Relationships>", "".join(new) + "</Relationships>"), encoding="utf8")

    ct_path = work / "[Content_Types].xml"
    ct = ct_path.read_text(encoding="utf8")
    if 'Extension="png"' not in ct:
        ct = ct.replace("<Default Extension=\"xml\"", '<Default Extension="png" ContentType="image/png"/><Default Extension="xml"')
    ct_path.write_text(ct, encoding="utf8")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ct_path, "[Content_Types].xml")
        for f in sorted(work.rglob("*")):
            if f.is_file() and f.name != "[Content_Types].xml":
                z.write(f, f.relative_to(work).as_posix())
    shutil.rmtree(work)
    print("wrote", OUT, "with", len(IMAGES), "images")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
