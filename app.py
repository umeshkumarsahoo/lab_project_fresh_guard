# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings("ignore")

# Flask app
app = Flask(__name__)

# Global cache for loaded artifacts (to avoid reloading on every request)
ARTIFACTS = {
    "model": None,
    "scaler": None,
    "encoders": None,
    "features": None,
    "numeric_cols": None,
    "categorical_cols": None
}

@app.route("/")
def home():
    # index.html should live in templates/index.html
    return render_template("index.html")


# ---------------------------
# Prediction route (form POST)
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict handler updated:
    - Brix (Initial_Brix_Level) and Firmness (Initial_Damage_Score) are optional for users.
      If missing, uses dataset mean values loaded from fresh_guard_preprocessed.csv.
    - Builds the full preprocessed feature vector expected by the model (one-hot Pretreatment_/Packaging_
      columns, Produce_Type target-enc, Produce_Type label if encoder available).
    """

    model_path = "best_regressor.pkl"
    scaler_path = "scaler.pkl"
    encoders_path = "encoders.pkl"
    metadata_path = "metadata.pkl"
    preproc_csv = "fresh_guard_preprocessed.csv"

    # Check artifacts
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoders_path)):
        msg = ("Model artifacts not found. Please run training first (run `python app.py` without --serve) "
               "so that best_regressor.pkl, scaler.pkl and encoders.pkl are created.")
        return render_template("index.html", result=f"Error: {msg}")

    # Lazy-load & cache artifacts
    if ARTIFACTS["model"] is None:
        ARTIFACTS["model"] = joblib.load(model_path)
    if ARTIFACTS["scaler"] is None:
        ARTIFACTS["scaler"] = joblib.load(scaler_path)
    if ARTIFACTS["encoders"] is None:
        ARTIFACTS["encoders"] = joblib.load(encoders_path)
    if ARTIFACTS["features"] is None and os.path.exists(metadata_path):
        meta = joblib.load(metadata_path)
        ARTIFACTS["numeric_cols"] = meta.get("numeric_cols")
        ARTIFACTS["categorical_cols"] = meta.get("categorical_cols")
        ARTIFACTS["features"] = meta.get("features")

    # Load or compute dataset means if not already cached
    if ARTIFACTS.get("means") is None:
        try:
            import pandas as _pd
            if os.path.exists(preproc_csv):
                _df = _pd.read_csv(preproc_csv)
                # compute means for numeric fields we may default to
                ARTIFACTS["means"] = {
                    "Initial_Brix_Level": float(_df["Initial_Brix_Level"].mean()) if "Initial_Brix_Level" in _df.columns else 0.0,
                    "Initial_Damage_Score": float(_df["Initial_Damage_Score"].mean()) if "Initial_Damage_Score" in _df.columns else 0.0
                }
            else:
                # fallback defaults
                ARTIFACTS["means"] = {"Initial_Brix_Level": 0.0, "Initial_Damage_Score": 0.0}
        except Exception:
            ARTIFACTS["means"] = {"Initial_Brix_Level": 0.0, "Initial_Damage_Score": 0.0}

    model = ARTIFACTS["model"]
    scaler = ARTIFACTS["scaler"]
    encoders = ARTIFACTS["encoders"]
    means = ARTIFACTS["means"]

    # Determine expected features (prefer model.feature_names_in_ if present)
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    else:
        expected_features = ARTIFACTS.get("features") or []
        # as last resort, try to infer from CSV
        if not expected_features:
            try:
                import pandas as _pd
                _df_tmp = _pd.read_csv(preproc_csv)
                expected_features = [c for c in _df_tmp.columns if c != "Remaining_Shelf_Life_Days"]
            except Exception:
                # extremely conservative fallback
                expected_features = ARTIFACTS.get("numeric_cols", []) + ARTIFACTS.get("categorical_cols", [])

    # mapping of produce -> target-encoded value (your provided map)
    produce_target_map = {
        "grape": 1.15200957,
        "banana": 0.76003577,
        "apple": 10.44604824,
        "potato": 16.78324158,
        "strawberry": 0.43322808,
        "tomato": 1.5981011,
        "orange": 6.1963276,
        "potto": 14.19429034,
        "ornge": 8.9242556,
        "tomto": 0.92158198,
        "graepe": 2.21592065,
        "aple": 0.0,
        "appel": 17.12188849,
        "stawberry": 1.60498671,
        "bananna": 2.08630067
    }

    form = request.form

    def get_form_val(*keys, default=None):
        for k in keys:
            v = form.get(k)
            if v is not None and v != "":
                return v
        return default

    try:
        # Initialize row dict with zeros
        row = {feat: 0.0 for feat in expected_features}

        # Fill numeric features using aliases; for Brix and Firmness use mean if missing
        numeric_aliases = {
            "Initial_Brix_Level": ["Initial_Brix_Level", "Brix", "brix"],
            "Initial_Weight": ["Initial_Weight", "Weight", "weight", "wt"],
            "Initial_Damage_Score": ["Initial_Damage_Score", "Damage_Score", "Damage", "Firmness", "firmness"],
            "Avg_Temp_C": ["Avg_Temp_C", "Temperature", "Temp", "temperature"],
            "Avg_Humidity_%": ["Avg_Humidity_%", "Humidity", "humidity", "Hum"],
            "Age_at_Measurement": ["Age_at_Measurement", "Age", "age"]
        }

        for feat in expected_features:
            if feat in numeric_aliases:
                val = None
                for alias in numeric_aliases[feat]:
                    raw = form.get(alias)
                    if raw not in (None, ""):
                        val = raw
                        break
                if val is None or val == "":
                    # Use dataset mean for Brix and Damage_Score specifically
                    if feat == "Initial_Brix_Level":
                        row[feat] = float(means.get("Initial_Brix_Level", 0.0))
                        continue
                    if feat == "Initial_Damage_Score":
                        row[feat] = float(means.get("Initial_Damage_Score", 0.0))
                        continue
                # parse float
                try:
                    row[feat] = float(val) if val is not None else 0.0
                except:
                    row[feat] = 0.0

        # Handle Produce_Type / Fruit -> set Produce_Type_Target_Enc & Produce_Type_Label
        prod_choice = get_form_val("Produce_Type", "Fruit", default=None)
        if prod_choice is None:
            prod_choice = "unknown"
        prod_choice = prod_choice.strip()

        if "Produce_Type_Target_Enc" in expected_features:
            row["Produce_Type_Target_Enc"] = float(produce_target_map.get(prod_choice, 0.0))

        if "Produce_Type_Label" in expected_features:
            if "Produce_Type" in encoders:
                le = encoders["Produce_Type"]
                if prod_choice in list(le.classes_):
                    row["Produce_Type_Label"] = int(le.transform([prod_choice])[0])
                else:
                    if "unknown" in list(le.classes_):
                        row["Produce_Type_Label"] = int(le.transform(["unknown"])[0])
                    else:
                        le.classes_ = np.append(le.classes_, prod_choice)
                        row["Produce_Type_Label"] = int(le.transform([prod_choice])[0])
            else:
                row["Produce_Type_Label"] = 0

        # Pretreatment and Packaging one-hot columns
        pret_val = get_form_val("Pretreatment", default="unknown")
        pack_val = get_form_val("Packaging", default="unknown")
        for feat in expected_features:
            if feat.startswith("Pretreatment_"):
                suffix = feat.split("Pretreatment_")[1]
                row[feat] = 1.0 if pret_val is not None and str(pret_val).lower() == suffix.lower() else 0.0
            if feat.startswith("Packaging_"):
                suffix = feat.split("Packaging_")[1]
                row[feat] = 1.0 if pack_val is not None and str(pack_val).lower() == suffix.lower() else 0.0

        # If any expected feature present directly in form, attempt to use its value (safe fallback)
        for feat in expected_features:
            v = form.get(feat)
            if v not in (None, ""):
                try:
                    row[feat] = float(v)
                except:
                    # keep existing value
                    pass

        # Build DataFrame in correct order
        import pandas as _pd
        df_in = _pd.DataFrame([row], columns=expected_features)

        # Ensure numeric dtype
        for c in df_in.columns:
            try:
                df_in[c] = df_in[c].astype(float)
            except:
                df_in[c] = 0.0

        # Apply scaler (scaler fitted on full training feature set)
        X_scaled = scaler.transform(df_in.values)

        # Predict
        pred = model.predict(X_scaled)[0]
        pred = float(pred)

        return render_template("index.html", result=f"Predicted Shelf Life: {pred:.2f} days")

    except Exception as e:
        return render_template("index.html", result=f"Prediction error: {str(e)}")


# ---------------------------
# Training pipeline (your original function)
# ---------------------------
def run_pipeline():
    # All data loading, processing and training happens inside this function.
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    # Update path to your dataset
    df = pd.read_csv("Datasets/FreshGuard_RAW_dirty.csv")

    print(df.head())
    print(df.info())
    print("\nMissing values:\n", df.isnull().sum())

    print("\nSTEP 2: FEATURE UNDERSTANDING")
    print("=" * 80)
    print("\nColumns in dataset:")
    print(df.columns)

    print("\nSTEP 3: EDA")
    print("=" * 80)

    target_col = "Remaining_Shelf_Life_Days"
    plt.figure(figsize=(10, 4))
    sns.histplot(df[target_col], kde=True)
    plt.title("Remaining Shelf Life Distribution")
    plt.savefig("shelf_life_distribution.png")

    print("\nSTEP 4: DATA CLEANING")
    print("=" * 80)
    df = df.drop_duplicates()
    df = df.dropna(subset=[target_col])
    print("Shape after cleaning:", df.shape)

    print("\nSTEP 5: FEATURE ENGINEERING")
    print("=" * 80)

    categorical_cols = [c for c in ["Produce_Type", "Pretreatment", "Packaging"] if c in df.columns]
    in_cols = ["Initial_Brix_Level", "Initial_Weight", "Initial_Damage_Score", "Avg_Temp_C", "Avg_Humidity_%", "Age_at_Measurement"]
    numeric_cols = [c for c in in_cols if c in df.columns]

    # Fill missing values for categorical features
    encoders = {}
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # For numeric columns, fill missing with median
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    features = numeric_cols + categorical_cols
    X = df[features]
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save preprocessing artifacts
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoders, "encoders.pkl")

    print("Preprocessing saved.")

    print("\nSTEP 6: TRAIN-TEST SPLIT")
    print("=" * 80)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("\nSTEP 7: MODEL TRAINING")
    print("=" * 80)
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        results[name] = (mae, rmse, r2)
        print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

    print("\nSTEP 8: SELECTING BEST MODEL")
    print("=" * 80)
    best_model_name = min(results, key=lambda k: results[k][1])
    best_model = models[best_model_name]
    joblib.dump(best_model, "best_regressor.pkl")

    print(f"Best Model: {best_model_name}")
    print("Saved as best_regressor.pkl")

    print("\nSTEP 9: ERROR ANALYSIS")
    print("=" * 80)
    final_pred = best_model.predict(X_test)
    error_df = pd.DataFrame({"Actual": y_test.values, "Predicted": final_pred, "Error": y_test.values - final_pred})
    print(error_df.head())
    plt.figure(figsize=(8, 4))
    sns.histplot(error_df["Error"], kde=True)
    plt.title("Prediction Error Distribution")
    plt.savefig("error_distribution.png")

    # Save metadata for the serving step (optional but helpful)
    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "features": features
    }
    joblib.dump(metadata, "metadata.pkl")

    # Save pipeline function is not necessary (we save artifacts separately)
    print("\nFINAL SUMMARY")
    print("=" * 80)
    print(f"Best model: {best_model_name}")
    print("Training complete. Artifacts saved successfully.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true', help='Run Flask server instead of training')
    args = parser.parse_args()

    if args.serve:
        # Optionally load metadata into ARTIFACTS so the predict route knows columns
        if os.path.exists("metadata.pkl"):
            meta = joblib.load("metadata.pkl")
            ARTIFACTS["numeric_cols"] = meta.get("numeric_cols")
            ARTIFACTS["categorical_cols"] = meta.get("categorical_cols")
            ARTIFACTS["features"] = meta.get("features")
        app.run(debug=True, host='0.0.0.0', port=8080)
    else:
        run_pipeline()
