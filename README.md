# Smart Community Health Monitoring and Early Warning System

## What's here

- `app/streamlit_app.py` — the Streamlit dashboard (Predict Risk + Data Explorer tabs,
  Combined Risk Summary using the OR/max-severity rule).
- `models/outbreak_model.joblib` — Module B.2, the final disease-outbreak risk model
  (EasyEnsemble, monthly-lagged features). Accuracy 76.2%, F1 0.379, ROC-AUC 0.766.
- `models/potability_submodel.joblib` — Module B.1, the water-contamination risk model
  (Random Forest on the Kaggle potability dataset). Accuracy 66.8%, F1 0.452.
- `data/processed/model_dataset.csv` — the merged, district/time-aligned IDSP + weather
  panel the dashboard reads for historical lookups and the Data Explorer tab.
- `data/processed/district_neighbor_map.json` — neighbor-district lookup used by one of
  the outbreak model's features (`neighbor_outbreak_recent`).
- `data/raw/` — the original source files (IDSP outbreak records, weather features,
  district centroids, Kaggle water-potability dataset), kept so `01_preprocess.py`
  and the two training scripts can be re-run end-to-end if needed.
- `src/01_preprocess.py` — builds `data/processed/model_dataset.csv` from the raw files.
- `src/02_train_potability_submodel.py` — trains Module B.1 and saves
  `models/potability_submodel.joblib`.
- `src/03_train_outbreak_model.py` — trains Module B.2 (the monthly-lagged,
  GridSearchCV-compared, EasyEnsemble-selected final model) and saves
  `models/outbreak_model.joblib`. This is Iteration 3.
- `requirements.txt` — exact package list needed.

## Running the dashboard

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The two `.joblib` models are already trained, so the dashboard runs immediately —
you don't need to re-run the `src/` scripts unless you want to retrain from scratch.
