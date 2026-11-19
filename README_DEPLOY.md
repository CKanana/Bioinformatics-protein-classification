# Demo app: Kinase or transcription factor?

Files to add to the GitHub repository (repo root):

    app.py
    requirements.txt          (replace the existing one; it adds streamlit)
    .streamlit/config.toml
    models/protein_family_model.joblib      (created by the Colab save-model cell)
    examples.json                           (created by the same cell)

## 1. Create the model file (in Colab, after the model comparison cell)
Run the contents of `save_model_cell.py` as a new cell. It saves the best model and four example
sequences to your shared Drive folder (`models/` and the folder root).
Download `protein_family_model.joblib` and `examples.json` from Drive.

## 2. Put the files on GitHub
On GitHub: Add file > Upload files, drop the five items above into the repo root (keep the `models/`
folder name), commit to your branch, open a pull request and merge into `main`.
If the model file is over 25 MB, lower `n_estimators` to 100 in the save cell and run it again.

## 3. Deploy for free on Streamlit Community Cloud
1. Go to share.streamlit.io and sign in with GitHub.
2. New app > pick the repository, branch `main`, main file path `app.py`.
3. Deploy. In about two minutes you get a public link to put on your slides.

## Run it locally instead (backup for the presentation)
    pip install -r requirements.txt
    streamlit run app.py
