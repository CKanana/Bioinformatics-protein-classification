# Protein Family Classification: Kinase vs Transcription Factor

Bioinformatics group project. We test whether simple physicochemical properties of amino acid sequences can predict if a human protein is a **kinase** or a **transcription factor**.

## Research question

Can physicochemical properties of amino acid sequences (such as hydrophobicity, molecular weight, and charge) accurately predict whether a set of proteins belongs to the kinase or transcription factor family?

## Objectives

1. Retrieve human protein sequence datasets for kinases and transcription factors from UniProtKB.
2. Extract numerical physicochemical features (amino acid composition, isoelectric point, molecular weight, GRAVY index) using Biopython.
3. Perform exploratory data analysis (PCA and feature correlation heatmaps).
4. Train, evaluate and interpret supervised machine learning models (Logistic Regression, Random Forest, SVM) against a baseline model.

## Team and responsibilities

| Stage | Member |
|---|---|
| Research question and data acquisition | Crystal Kanana |
| Preprocessing and quality control | Andrew Kigondu |
| Exploratory data analysis | Emmanuel Douglas Ouma |
| Core analysis and biological interpretation | Gloria Kendi |
| Machine learning and conclusions | Donell Bikketi and Naomi Teko |

## Workflow

- **Google Colab** is where the work is done, in the shared notebook `notebooks/bioinformatics_project.ipynb`.
- **Google Drive** stores the dataset (`Bioinformatics_Project/data`) so everyone works from the same files.
- **GitHub** (this repository) stores the code and results. Analysis outputs are pushed here from Colab.
- The best model is deployed after evaluation.

## Repository structure

```
notebooks/   shared Colab notebook
data/        raw/ and processed/ (large data files stay on Google Drive)
results/     figures/ and tables/ pushed from Colab
models/      saved models
```

## How to run

1. Open `notebooks/bioinformatics_project.ipynb` in Google Colab (File > Open notebook > GitHub tab).
2. Run the setup cell. It mounts Google Drive and installs the dependencies.
3. Run the cells for your stage only. Earlier stages write their output to Google Drive for the next stage to read.

## Technologies

Python, Biopython, pandas, NumPy, scikit-learn, matplotlib, seaborn, UniProt REST API, Google Colab, GitHub.

## Git rules for the team

- Do not commit directly to `main`. Create your own branch, for example `crystal-data` or `donell-models`, and open a pull request.
- Commit small and often, with a clear message such as `Add preprocessing QC step`.
- Commit result files (figures, tables) rather than editing someone else's notebook section. Notebooks merge badly, so tell the group before editing the shared notebook.
- Never commit tokens or passwords.

## Timeline

| Task | Deadline |
|---|---|
| Colab, Drive and GitHub set up | 5 Nov 2026 |
| Data acquisition and preprocessing | 6 Nov 2026 |
| Exploratory analysis | 8 Nov 2026 |
| Model training and comparison | 9 to 15 Nov 2026 |
| Deployment and testing | by 18 Nov 2026 |
| Presentations begin | 19 Nov 2026 |
