import json, joblib
from sklearn.base import clone

# 1. The final model = the tuned model with the best held-out F1 (your tuned SVM).
#    To pick another one by hand, replace the next line, e.g. BEST = 'Random Forest'
BEST = test_results.iloc[0]['model']
print('Saving model:', BEST)

final = clone(best_models[BEST])           # tuned settings from the grid search
if BEST == 'SVM (RBF)':
    final.set_params(svc__probability=True, svc__random_state=42)   # the app needs probabilities
final.fit(X, y)                            # refit on all cleaned proteins

# 2. Everything the app needs, in one file
pred = test_preds[BEST]
yt = y_test.values
main = ['length', 'mol_weight', 'pI', 'gravy', 'charge_pH7']
bundle = {
    'model': final,
    'model_name': BEST + ' (tuned)',
    'feature_cols': FEATURE_COLS,
    'n_kinase': int((y == 1).sum()),
    'n_tf': int((y == 0).sum()),
    'train_length_range': (float(feat['length'].min()), float(feat['length'].max())),
    'medians': feat.groupby('family')[main].median().to_dict('index'),
    'test_metrics': test_results.round(3).to_dict('list'),
    'kinase_miss_rate': float(((pred == 0) & (yt == 1)).sum() / (yt == 1).sum()),
    'tf_false_alarm_rate': float(((pred == 1) & (yt == 0)).sum() / (yt == 0).sum()),
}
os.makedirs(MODELS, exist_ok=True)
joblib.dump(bundle, f'{MODELS}/protein_family_model.joblib')

# 3. Four real example sequences for the app's buttons (two per family, from the cleaned data)
examples = []
for fam, name in [('kinase', 'Kinase'), ('transcription_factor', 'Transcription factor')]:
    pick = clean[clean['family'] == fam].sample(2, random_state=7)
    for i, (_, r) in enumerate(pick.iterrows(), 1):
        examples.append({'label': f'{name} {i}', 'sequence': r['Sequence']})
json.dump(examples, open(f'{BASE}/examples.json', 'w'))

size_mb = os.path.getsize(f'{MODELS}/protein_family_model.joblib') / 1e6
print(f'Saved {MODELS}/protein_family_model.joblib ({size_mb:.1f} MB) and {BASE}/examples.json')
