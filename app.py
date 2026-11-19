"""Kinase or transcription factor? A demo of the Group 6 bioinformatics project.

Loads a trained model bundle saved from the Colab notebook, turns a pasted protein
sequence into the same physicochemical features used in training, and shows the
predicted family with a probability. Nothing the user enters is stored.
"""
import json
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from Bio.SeqUtils.ProtParam import ProteinAnalysis

try:
    ROOT = Path(__file__).resolve().parent
except NameError:  # pragma: no cover
    ROOT = Path.cwd()

MODEL_PATH = ROOT / "models" / "protein_family_model.joblib"
EXAMPLES_PATH = ROOT / "examples.json"
FIGURES_DIR = ROOT / "results" / "figures"

AA = list("ACDEFGHIKLMNPQRSTVWY")
KINASE, TF = 1, 0
UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.40, 0.60
MAIN_PROPS = {
    "length": "Length (residues)",
    "mol_weight": "Molecular weight (Da)",
    "pI": "Isoelectric point (pI)",
    "gravy": "Hydrophobicity (GRAVY)",
    "charge_pH7": "Charge at pH 7",
}

st.set_page_config(page_title="Kinase or transcription factor?", page_icon="🧬", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Serif+Display&display=swap');
html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 2rem; max-width: 880px; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }

.hero { background: linear-gradient(135deg,#0B3C5D 0%,#0F766E 100%); color:#fff;
        border-radius: 20px; padding: 34px 36px 30px; margin-bottom: 14px; position:relative; overflow:hidden; }
.hero:after { content:""; position:absolute; right:-60px; top:-60px; width:240px; height:240px;
        border-radius:50%; background:rgba(255,255,255,.07); }
.hero .tag { font-size:.78rem; letter-spacing:.14em; text-transform:uppercase; opacity:.8; }
.hero h1 { font-family:'DM Serif Display',serif; font-weight:400; font-size:2.5rem; line-height:1.1; margin:.35rem 0 .6rem; color:#fff; padding:0; }
.hero p { font-size:.98rem; opacity:.92; max-width:640px; margin:0; line-height:1.5; }

.notice { background:#FFF8E6; border:1px solid #F4D58D; color:#6B4E00; border-radius:12px;
          padding:10px 16px; font-size:.86rem; margin-bottom:10px; }

.stTabs [data-baseweb="tab-list"] { gap: 18px; }
.stTabs button[role="tab"] p { font-size:1rem; font-weight:500; color:#4A5D70; }
.stTabs button[role="tab"][aria-selected="true"] p { color:#0F766E; font-weight:700; }

h2, h3 { font-family:'DM Serif Display',serif; font-weight:400; color:#0B3C5D; }
.stTextArea textarea { font-family: ui-monospace, Menlo, Consolas, monospace; font-size:.9rem;
        border-radius:14px; background:#F7FAFC; border:1.5px solid #D5E3EE; }
.stTextArea textarea:focus { border-color:#0F766E; }
.stButton > button { border-radius:999px; border:1.5px solid #CFE0EA; padding:.35rem 1.1rem; font-weight:500; }
.stButton > button:hover { border-color:#0F766E; color:#0F766E; }
.stButton > button[kind="primary"] { background:#0F766E; color:#fff; border:none; padding:.6rem 1.8rem; font-weight:700; }
.stButton > button[kind="primary"]:hover { background:#0B5E58; color:#fff; }

.result { border-radius:18px; padding:22px 26px; margin:18px 0 8px; border:1.5px solid; }
.result.kin { background:#E8F5F3; border-color:#7CC4BA; }
.result.tf  { background:#EAF0FB; border-color:#9DB7E8; }
.result.unc { background:#FFF4E0; border-color:#F0C27B; }
.result .lbl { font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; color:#4A5D70; }
.result .big { font-family:'DM Serif Display',serif; font-size:2rem; color:#0B3C5D; margin:.1rem 0 .2rem; }
.result .sub { color:#3C4F63; font-size:.95rem; }

.pbar { display:flex; height:30px; border-radius:999px; overflow:hidden; margin:14px 0 6px; font-size:.82rem; font-weight:700; color:#fff; }
.pbar .k { background:#0F766E; display:flex; align-items:center; padding-left:14px; white-space:nowrap; }
.pbar .t { background:#3B6FD4; display:flex; align-items:center; justify-content:flex-end; padding-right:14px; white-space:nowrap; }
.legend { display:flex; justify-content:space-between; font-size:.8rem; color:#4A5D70; }

.stat-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:8px 0 18px; }
.stat { background:#F4F8FB; border-radius:14px; padding:14px 16px; }
.stat .n { font-family:'DM Serif Display',serif; font-size:1.7rem; color:#0F766E; }
.stat .d { font-size:.8rem; color:#4A5D70; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- features: must match the notebook exactly ----------
def features(seq: str) -> dict:
    pa = ProteinAnalysis(seq)
    d = {f"aa_{a}": seq.count(a) / len(seq) for a in AA}
    d.update(
        length=len(seq),
        mol_weight=pa.molecular_weight(),
        pI=pa.isoelectric_point(),
        gravy=pa.gravy(),
        charge_pH7=pa.charge_at_pH(7.0),
        aromaticity=pa.aromaticity(),
        instability=pa.instability_index(),
    )
    return d


def clean_sequence(raw: str) -> str:
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith(">")]
    return re.sub(r"[\s\d]", "", "".join(lines)).upper()


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_examples():
    if EXAMPLES_PATH.exists():
        try:
            return json.loads(EXAMPLES_PATH.read_text())
        except Exception:
            return []
    return []


bundle = load_bundle()

# ---------- header ----------
st.markdown(
    """
<div class="hero">
  <div class="tag">Group 6 · Bioinformatics · Strathmore University</div>
  <h1>Kinase or transcription factor?</h1>
  <p>Paste a protein sequence and see whether its physicochemical profile looks like a protein kinase
  or a transcription factor.</p>
</div>
<div class="notice"><b>Research use only.</b> A prediction is a statistical guess from sequence properties.
Check it against experiments or a trusted database. Never use it for clinical, insurance or employment decisions.
Sequences you enter are not stored.</div>
""",
    unsafe_allow_html=True,
)
with st.expander("Our research question"):
    st.write(
        "Can physicochemical properties of amino acid sequences such as hydrophobicity, molecular weight "
        "and charge accurately predict whether a set of proteins belongs to the kinase or transcription factor family?"
    )

if bundle is None:
    st.error(
        "The trained model file was not found. Save it from the Colab notebook to "
        "`models/protein_family_model.joblib` in this repository, then reload the app."
    )
    st.stop()

model = bundle["model"]
feature_cols = bundle["feature_cols"]

tab_predict, tab_model, tab_limits = st.tabs(["Predict", "About the model", "Limits and ethics"])

# ---------- tab 1: predict ----------
with tab_predict:
    st.subheader("Paste a protein sequence")
    st.write("Plain letters or FASTA format. Only the 20 standard amino acids are accepted.")

    examples = load_examples()
    if examples:
        st.write("Or try a real example from our dataset (these come from the training data, so the model has seen them):")
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            if col.button(ex["label"], key="ex_" + ex["label"]):
                st.session_state["seq"] = ex["sequence"]

    raw = st.text_area("Amino acid sequence", key="seq", height=160, placeholder="MTVKTEAAKGTLTYSRMRG...")
    go = st.button("Predict family", type="primary")

    if go:
        seq = clean_sequence(raw)
        bad = sorted(set(seq) - set(AA))
        if not seq:
            st.warning("Please paste a sequence first.")
        elif bad:
            st.error(
                "The sequence contains characters that are not standard amino acids: "
                + ", ".join(bad)
                + ". Remove or correct them and try again."
            )
        elif len(seq) < 30:
            st.warning("That sequence is very short (under 30 residues). Predictions are unreliable.")
        else:
            row = features(seq)
            X = pd.DataFrame([row])[feature_cols]
            proba = model.predict_proba(X)[0]
            p_kin = float(proba[list(model.classes_).index(KINASE)])
            p_tf = 1.0 - p_kin

            if UNCERTAIN_LOW <= p_kin <= UNCERTAIN_HIGH:
                cls, lbl, big = "unc", "Result", "Uncertain"
                sub = f"The model puts the chance of a kinase at {p_kin:.0%}. That is too close to call, so treat it as no prediction."
            elif p_kin > UNCERTAIN_HIGH:
                cls, lbl, big = "kin", "Predicted family", "Protein kinase"
                sub = f"{p_kin:.0%} probability"
            else:
                cls, lbl, big = "tf", "Predicted family", "Transcription factor"
                sub = f"{p_tf:.0%} probability"
            st.markdown(
                f'<div class="result {cls}"><div class="lbl">{lbl}</div><div class="big">{big}</div>'
                f'<div class="sub">{sub}</div>'
                f'<div class="pbar"><div class="k" style="width:{max(p_kin*100,14):.1f}%">Kinase {p_kin:.0%}</div>'
                f'<div class="t" style="width:{max(p_tf*100,14):.1f}%">TF {p_tf:.0%}</div></div>'
                f'<div class="legend"><span>Protein kinase</span><span>Transcription factor</span></div></div>',
                unsafe_allow_html=True,
            )

            lo, hi = bundle.get("train_length_range", (None, None))
            if lo is not None and not (lo <= len(seq) <= hi):
                st.warning(
                    f"This sequence has {len(seq)} residues, outside the {lo:.0f} to {hi:.0f} range "
                    "the model was trained on, so treat the result with extra caution."
                )

            st.subheader("Your sequence compared with the training data")
            med = bundle.get("medians", {})
            table = []
            for key, label in MAIN_PROPS.items():
                table.append(
                    {
                        "Property": label,
                        "Your sequence": round(row[key], 3),
                        "Typical kinase": round(med.get("kinase", {}).get(key, float("nan")), 3),
                        "Typical transcription factor": round(
                            med.get("transcription_factor", {}).get(key, float("nan")), 3
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(table), hide_index=True, width='stretch')
            st.caption("Typical = the median in the training data. This shows what the model was given, not why it decided.")

# ---------- tab 2: about the model ----------
with tab_model:
    st.subheader("What the model is")
    n_k = bundle.get("n_kinase")
    n_t = bundle.get("n_tf")
    st.write(
        f"**Model:** {bundle.get('model_name', 'unknown')}  \n"
        f"**Training data:** reviewed human proteins from UniProtKB"
        + (f", {n_k} protein kinases and {n_t} transcription factors" if n_k and n_t else "")
        + "  \n**Inputs:** amino acid composition, length, molecular weight, isoelectric point, "
        "hydrophobicity (GRAVY), charge at pH 7, aromaticity and instability index."
    )
    if n_k and n_t:
        base = n_t / (n_k + n_t)
        st.write(
            f"The classes are imbalanced, so a model that always answers 'transcription factor' "
            f"would be right about {base:.0%} of the time. That is the baseline our models must beat, "
            "which is why we report F1, recall on kinases and ROC-AUC and not accuracy alone."
        )

    tm = bundle.get("test_metrics")
    if tm and "f1" in tm:
        i = 0
        cards = ""
        for key, name in [("accuracy", "Accuracy"), ("recall", "Kinase recall"), ("f1", "F1 score"), ("roc_auc", "ROC-AUC")]:
            if key in tm:
                cards += f'<div class="stat"><div class="n">{tm[key][i]:.2f}</div><div class="d">{name}</div></div>'
        st.markdown(f'<div class="stat-row">{cards}</div>', unsafe_allow_html=True)
    cv = bundle.get("cv_metrics")
    if tm:
        st.write("**Held-out test results** (20% of the proteins, kept aside and scored once, after tuning):")
        st.dataframe(pd.DataFrame(tm), hide_index=True, width='stretch')
    elif cv:
        st.write("**Cross-validated results** (5-fold, stratified):")
        st.dataframe(pd.DataFrame(cv), hide_index=True, width='stretch')

    figs = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    if figs:
        st.write("**Project figures**")
        for fig in figs:
            st.image(str(fig), caption=fig.stem.replace("_", " ").capitalize(), width='stretch')

# ---------- tab 3: limits and ethics ----------
with tab_limits:
    st.subheader("Known limits")
    miss = bundle.get("kinase_miss_rate")
    fa = bundle.get("tf_false_alarm_rate")
    if miss is not None and fa is not None:
        st.warning(
            f"On our held-out test set the model missed {miss:.0%} of kinases and wrongly called "
            f"{fa:.0%} of transcription factors kinases. A 'transcription factor' answer for a real "
            "kinase is the most common mistake."
        )
    st.markdown(
        """
- **Who is missing from the data.** Reviewed entries favour well-studied proteins. Poorly studied
  kinases and transcription factors are under-represented, and reference sequences do not capture
  variation across populations, including African populations.
- **Labels come from earlier annotation.** Kinase and transcription factor labels come from UniProt
  keywords and Gene Ontology terms, which can be wrong or incomplete. Proteins labelled as both were removed.
- **Only two families.** Any protein that is neither a kinase nor a transcription factor will still
  be given one of the two answers, so the output is meaningless for other protein types.
- **Not peer reviewed.** This is exploratory student work.
- **Properties, not mechanism.** The model uses overall sequence properties, not domains or motifs.
"""
    )
    st.subheader("Privacy")
    st.write(
        "The app accepts a protein sequence only. It stores nothing you enter and has no fields "
        "for personal or patient data. Please do not paste sequences from identifiable patients."
    )
    st.subheader("Credits")
    st.write(
        "Data: UniProt Consortium, UniProtKB (reviewed human entries). "
        "Tools: Biopython, scikit-learn, Streamlit."
    )