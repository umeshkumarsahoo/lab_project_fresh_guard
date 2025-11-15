**FreshGuard**

FreshGuard is a small ML project for predicting remaining shelf life of produce using a supervised regression pipeline and a lightweight Flask front-end for quick predictions.

**Quick Overview**
- **Project:** Predict remaining shelf life in days for produce items.
- **Data:** Raw dataset is in `Datasets/FreshGuard_RAW_dirty.csv`.
- **Model artifacts:** `best_regressor.pkl`, `scaler.pkl`, `encoders.pkl`, `metadata.pkl`, `prediction_pipeline.pkl` are produced by the training pipeline.

**Requirements**
- Python 3.10+ (the project was developed with Python 3.12 compatible code).
- See `requirements.txt` for required packages.

**Setup (macOS / zsh)**
1. Create a virtual environment and activate it (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) If using VS Code, select the `.venv` interpreter: Command Palette → `Python: Select Interpreter` → choose `<project>/ .venv/bin/python`.

**Run the training pipeline**
- This will load `Datasets/FreshGuard_RAW_dirty.csv`, preprocess, train multiple models, select the best model, and write artifacts to disk.

```bash
# from project root
source .venv/bin/activate
python app.py
```

Artifacts created by training (examples):
- `best_regressor.pkl` — selected trained model
- `scaler.pkl` — StandardScaler used for numeric features
- `encoders.pkl` — dictionary of LabelEncoder objects for categorical columns
- `metadata.pkl` — list of `numeric_cols`, `categorical_cols`, and `features`
- `prediction_pipeline.pkl` — convenience saved pipeline (if produced)

**Run the Flask server (serve only, no training)**
- Start the web UI (the server serves `index.html` from the project root by default) on port 8080:

```bash
# serve only (does NOT retrain)
source .venv/bin/activate
python app.py --serve
```

Open your browser to: `http://127.0.0.1:8080`

**Prediction endpoint**
- The UI form posts to `/predict` and returns a prediction rendered on the page.
- The server expects saved artifacts (`best_regressor.pkl`, `scaler.pkl`, `encoders.pkl`, `metadata.pkl`) produced by the training run. If they are missing, the page will display an error advising to run training first.

Example curl (fields taken from `metadata.pkl` — numeric followed by categorical):

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -F "Initial_Brix_Level=15.2" \
  -F "Initial_Weight=120.5" \
  -F "Initial_Damage_Score=2.3" \
  -F "Avg_Temp_C=4.5" \
  -F "Avg_Humidity_%=78" \
  -F "Age_at_Measurement=3" \
  -F "Produce_Type=apple" \
  -F "Pretreatment=none" \
  -F "Packaging=loose"
```

**Dataset**
- Raw dataset used for training: `Datasets/FreshGuard_RAW_dirty.csv`.
- A preprocessed version `fresh_guard_preprocessed.csv` may be created/used by local scripts; the code will fallback to safe defaults if missing.

**Development notes**
- The training pipeline is implemented in `app.py` in `run_pipeline()`; serving the web UI is done by running `python app.py --serve`.
- The server finds `index.html` in the project root (the code sets `Flask(..., template_folder='.')`) so you can keep the HTML file in the repo root.
- If you prefer the standard Flask layout, move the HTML into `templates/index.html`.

**Valid categorical values (inferred from saved encoders)**
The server uses saved LabelEncoders for the categorical columns. These are the values the encoder was trained on and which the server expects (case-sensitive exact strings). If an unseen value is provided, the server attempts fallbacks but it's best to use one of the known values below:

- **Produce_Type**:
  - aple, appel, apple, banana, bananna, graepe, grape, orange, ornge, potato, potto, stawberry, strawberry, tomato, tomto
- **Pretreatment**:
  - none, pre-cooled, unknown, washed, waxed
- **Packaging**:
  - carton, loose, perforated_bag, sealed_punnet, unknown

Use these exact strings when posting to `/predict` or filling the form.

**Makefile examples**
The project includes a `Makefile` with convenient targets. Examples:

- Install dependencies and create the virtual environment:

```bash
make install
```

- Run the full training pipeline (creates model artifacts):

```bash
make train
```

- Start the Flask server on port 8080 (serve only, no retraining):

```bash
make serve
```

- Clean generated artifacts (models, pickles, images):

```bash
make clean
```

- Show concise git status:

```bash
make status
```

These commands call the `.venv` Python/pip binaries directly so they work consistently across shells.

**Contributors**
- Prachi
- Devi
- Sneha
- Umesh

**License**
- See the `LICENSE` file in the project root for license details.
