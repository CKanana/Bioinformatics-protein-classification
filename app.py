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

st.set_page_config(page_title="Kinase or transcription factor?", layout="wide")


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
st.title("Kinase or transcription factor?")
st.caption(
    "Group 6 bioinformatics project | Strathmore University | "
    "Can physicochemical properties of amino acid sequences such as hydrophobicity, molecular weight "
    "and charge accurately predict whether a set of proteins belongs to the kinase or transcription factor family?"
)
st.info(
    "For learning and research use only. A prediction is a statistical guess from sequence "
    "properties and must be checked against experiments or a trusted database. "
    "Never use it for clinical, insurance or employment decisions. "
    "Sequences you enter are not stored."
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
                st.warning(
                    f"Uncertain. The model puts the probability of a kinase at {p_kin:.0%}, "
                    "which is too close to call. Treat this as no prediction."
                )
            elif p_kin > UNCERTAIN_HIGH:
                st.success(f"Predicted family: protein kinase ({p_kin:.0%} probability)")
            else:
                st.success(f"Predicted family: transcription factor ({p_tf:.0%} probability)")

            c1, c2 = st.columns(2)
            c1.metric("Probability: protein kinase", f"{p_kin:.0%}")
            c2.metric("Probability: transcription factor", f"{p_tf:.0%}")
            st.bar_chart(
                pd.DataFrame(
                    {"Probability": [p_kin, p_tf]},
                    index=["Protein kinase", "Transcription factor"],
                )
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
