
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from scipy import sparse

warnings.filterwarnings("ignore")
st.set_page_config(page_title="CreditPath AI — Combined", layout="wide")

ROOT = Path(".").resolve()
DATASETS = {
    "Microsoft": {
        "models": ROOT / "microsoft_dataset" / "models",
        "data": ROOT / "microsoft_dataset" / "data"
    },
    "Kaggle": {
        "models": ROOT / "kaggle_dataset" / "models",
        "data": ROOT / "kaggle_dataset" / "data"
    }
}

MICRO_EXPECTED = [
    "memberId","residentialState","yearsEmployment","homeOwnership","annualIncome","incomeVerified",
    "dtiRatio","lengthCreditHistory","numTotalCreditLines","numOpenCreditLines","numOpenCreditLines1Year",
    "revolvingBalance","revolvingUtilizationRate","numDerogatoryRec","numDelinquency2Years","numChargeoff1year","numInquiries6Mon","loanId","purpose","isJointApplication","loanAmount","term","interestRate","monthlyPayment","grade","year","month"
]


KAGGLE_EXPECTED = [
    'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate', 'installment', 'annual_inc', 'loan_status', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'last_pymnt_amnt', 'emp_length_years', 'home_ownership_1', 'home_ownership_2', 'home_ownership_3', 'home_ownership_4', 'purpose_1', 'purpose_2', 'purpose_3', 'purpose_4', 'purpose_5', 'purpose_6', 'purpose_7', 'purpose_8', 'purpose_9', 'purpose_10', 'purpose_11', 'purpose_12', 'purpose_13', 'verification_status_1', 'verification_status_2'
]


TARGET_COL = "defaultFlag"

# ---------- utilities ----------
def list_models(models_dir):
    if not models_dir.exists():
        return []
    return [p for p in sorted(models_dir.iterdir()) if p.suffix.lower() in (".joblib", ".pkl")]

def force_to_expected(df_in, expected_features):
    df = df_in.copy()
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = 0
    df = df[expected_features].copy()
    df = df.fillna(0)
    return df

def prepare_array(X_df, scaler=None):
    X = X_df.copy().fillna(0)
    try:
        arr = X.values.astype(float)
    except Exception:
        arr = X.values
    if scaler is not None:
        try:
            arr = scaler.transform(arr)
        except Exception:
            pass
    return arr

def get_positive_probs(mobj, arr):
    if not hasattr(mobj, "predict_proba"):
        return None
    probs_all = mobj.predict_proba(arr)
    classes = getattr(mobj, "classes_", None)
    if classes is not None:
        try:
            pos_idx = list(classes).index(1)
        except ValueError:
            pos_idx = probs_all.shape[1] - 1
    else:
        pos_idx = probs_all.shape[1] - 1
    return probs_all[:, pos_idx]

def coerce_grade_column(df, models_dir):
    # simple attempt: if grade exists and is non-numeric, encode deterministically
    if "grade" not in df.columns:
        return df
    try:
        df["grade"] = pd.to_numeric(df["grade"], errors="raise")
        return df
    except Exception:
        # build mapping from sample if present
        data_base = models_dir.parent / "data"
        mapping = None
        if data_base.exists():
            files = list(data_base.rglob("*.csv"))
            for p in files:
                if "03_processed" in str(p).replace("\\","/"):
                    try:
                        sample = pd.read_csv(p, nrows=200)
                        if "grade" in sample.columns:
                            vals = sorted(sample["grade"].dropna().unique().tolist())
                            mapping = {v: i+1 for i,v in enumerate(vals)}
                            break
                    except Exception:
                        continue
        # fallback mapping from provided df
        if mapping is None:
            vals = sorted(df["grade"].dropna().unique().tolist())
            mapping = {v: i+1 for i,v in enumerate(vals)} if vals else None
        if mapping:
            df["grade"] = df["grade"].map(mapping).fillna(0).astype(float)
        else:
            df["grade"] = 0.0
        return df

# ---------- Sidebar: dataset & controls ----------
st.sidebar.title("Dataset & Controls")
dataset_choice = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
models_dir = DATASETS[dataset_choice]["models"]
data_dir = DATASETS[dataset_choice]["data"]

EXPECTED = MICRO_EXPECTED if dataset_choice == "Microsoft" else KAGGLE_EXPECTED

model_files = list_models(models_dir)
if not model_files:
    st.sidebar.error(f"No model files in {models_dir}. Place .joblib/.pkl models there.")
    st.stop()

model_choice = st.sidebar.selectbox("Model", [p.stem for p in model_files])
model_path = next(p for p in model_files if p.stem == model_choice)

st.sidebar.markdown("---")
default_show = EXPECTED[:10]
selected_features = st.sidebar.multiselect("Single-record form fields", options=EXPECTED, default=default_show)
debug_toggle = st.sidebar.checkbox("Show model debug info", value=False)

# load scaler/preprocessor if present
scaler = None
scaler_path = models_dir / "scaler.joblib"
preproc_path = models_dir / "preprocessor_artifacts.joblib"
if scaler_path.exists():
    try:
        loaded = joblib.load(scaler_path)
        scaler = loaded.get("scaler", loaded) if isinstance(loaded, dict) else loaded
    except Exception:
        scaler = None

preprocessor = None
preproc_path = models_dir / "preprocessor_artifacts.joblib"

if preproc_path.exists():
    try:
        loaded = joblib.load(preproc_path)
        # If it's already a transformer, use directly
        if hasattr(loaded, "transform"):
            preprocessor = loaded
            if debug_toggle:
                st.sidebar.write("preprocessor_artifacts.joblib: loaded transformer directly.")
        elif isinstance(loaded, dict):
            # If dict: check keys
            if debug_toggle:
                st.sidebar.write(f"preprocessor_artifacts keys: {list(loaded.keys())}")

            # expected keys in your file: 'imputer', 'scaler', 'numeric_columns', 'feature_columns'
            imputer = loaded.get("imputer", None)
            scaler_obj = loaded.get("scaler", None)
            numeric_cols = loaded.get("numeric_columns", None)
            feature_cols = loaded.get("feature_columns", None)

            # if numeric_cols is Index convert to list of strings
            if isinstance(numeric_cols, (list, tuple, pd.Index)):
                numeric_cols_list = list(numeric_cols)
            else:
                numeric_cols_list = []

            # fallback feature_cols to EXPECTED if missing
            if feature_cols is None:
                feature_cols = EXPECTED  # safe fallback
                if debug_toggle:
                    st.sidebar.write("preprocessor_artifacts.joblib: feature_columns missing — using EXPECTED as fallback.")

            # Basic validation
            if imputer is not None and scaler_obj is not None and feature_cols is not None:
                # Build a small wrapper with transform()
                class PreprocessorWrapper:
                    def __init__(self, imputer, scaler, numeric_cols, feature_cols):
                        self.imputer = imputer
                        self.scaler = scaler
                        self.numeric_cols = list(numeric_cols) if numeric_cols is not None else []
                        self.feature_cols = list(feature_cols)

                    def transform(self, X_df):
                        """
                        X_df: pandas DataFrame (rows to transform)
                        Returns: 2D numpy array aligned to self.feature_cols order
                        """
                        X = X_df.copy()
                        # Ensure columns are strings and trimmed
                        X.columns = [str(c).strip() for c in X.columns]

                        # 1) Prepare an output DataFrame with all feature_cols filled with zeros
                        out = pd.DataFrame(0, index=X.index, columns=self.feature_cols, dtype=float)

                        # 2) If numeric columns present, apply imputer+scaler
                        if self.numeric_cols:
                            # select those that exist in X (missing will be filled with NaN for imputer)
                            numeric_present = [c for c in self.numeric_cols if c in X.columns]
                            # create a numeric matrix for imputer (order = self.numeric_cols)
                            num_for_impute = X.reindex(columns=self.numeric_cols)  # missing cols => NaN columns
                            # apply imputer (returns numpy array)
                            try:
                                imputed = self.imputer.transform(num_for_impute)
                            except Exception as e:
                                # fallback: attempt to fill na with 0 then proceed
                                num_for_impute = num_for_impute.fillna(0)
                                imputed = self.imputer.transform(num_for_impute)
                            # scale
                            try:
                                scaled = self.scaler.transform(imputed)
                            except Exception:
                                # if scaler fails (e.g., expects 2D), just use imputed
                                scaled = imputed
                            # place scaled numeric columns into out DataFrame where column names match numeric_cols
                            for idx, colname in enumerate(self.numeric_cols):
                                if colname in out.columns:
                                    out[colname] = scaled[:, idx]
                                else:
                                    # if numeric column isn't in feature_cols, skip
                                    continue

                        # 3) For non-numeric columns listed in feature_cols, copy values from X if exist
                        non_num_cols = [c for c in self.feature_cols if c not in self.numeric_cols]
                        for c in non_num_cols:
                            if c in X.columns:
                                # attempt to coerce to numeric; if fails, leave as-is and try to cast
                                vals = X[c].fillna(0)
                                # if all values are numeric-like, cast, else try to map bool/text to numeric
                                try:
                                    out[c] = pd.to_numeric(vals, errors="coerce").fillna(0).astype(float)
                                except Exception:
                                    # fallback: for strings, simple mapping via factorizing
                                    try:
                                        codes = pd.factorize(vals.astype(str))[0].astype(float)
                                        out[c] = codes
                                    except Exception:
                                        out[c] = 0.0
                            else:
                                # column not present in uploaded X -> leave zeros
                                out[c] = 0.0

                        # final: ensure numeric numpy array
                        return out.values

                preprocessor = PreprocessorWrapper(imputer=imputer, scaler=scaler_obj,
                                                  numeric_cols=numeric_cols_list, feature_cols=feature_cols)
                if debug_toggle:
                    st.sidebar.write("PreprocessorWrapper created: will apply imputer->scaler on numeric_cols and align to feature_columns.")
            else:
                if debug_toggle:
                    st.sidebar.write("preprocessor_artifacts.joblib loaded but missing required pieces (imputer/scaler/feature_columns).")
                preprocessor = None
        else:
            # unknown object type
            if debug_toggle:
                st.sidebar.write("preprocessor_artifacts.joblib loaded but no transform() found and not a dict.")
            preprocessor = None
    except Exception as e:
        st.sidebar.warning(f"Failed loading preprocessor_artifacts.joblib: {e}")
        preprocessor = None
else:
    preprocessor = None


# display
st.title(f"CreditPath AI — {dataset_choice}")
st.write("Model folder:", str(models_dir))
st.write("Using model:", model_choice)

# ---------- Single-record tab ----------
st.header("Single-record prediction")
# allow upload of 1-row CSV OR manual form (prefill from session if loaded)
uploaded_one = st.file_uploader(f"Upload 1-row processed CSV for {dataset_choice} (optional)", type=["csv"], key=f"one_{dataset_choice}")
if uploaded_one:
    one_df = pd.read_csv(uploaded_one)
    if dataset_choice == "Microsoft":
        one_df = coerce_grade_column(one_df, models_dir)
    X_single = force_to_expected(one_df, EXPECTED).iloc[[0]]
else:
    loaded = st.session_state.get("loaded_single_row", None)
    # build manual small form based on selected_features
    cols = st.columns(3)
    manual_input = {}
    for i, feat in enumerate(selected_features):
        col = cols[i % 3]
        if feat.lower() in ["memberid","loanid"] or any(x in feat.lower() for x in ["year","month","num","count","id"]):
            manual_input[feat] = col.number_input(feat, value=0)
        elif any(x in feat.lower() for x in ["amt","amount","income","rate","ratio","dti","loan","installment","payment","interest"]):
            manual_input[feat] = col.number_input(feat, value=0.0, format="%.6f")
        else:
            default_val = loaded.get(feat, "0") if loaded else "0"
            manual_input[feat] = col.text_input(feat, value=str(default_val))
    X_single = pd.DataFrame([manual_input])
    if dataset_choice == "Microsoft":
        X_single = coerce_grade_column(X_single, models_dir)
    X_single = force_to_expected(X_single, EXPECTED)

st.write("Preview (first 8 cols):")
st.dataframe(X_single.iloc[:, :8].T)

if st.button("Predict single"):
    try:
        # Load chosen model
        m = joblib.load(model_path)

        # Prepare input depending on dataset & model expectations
        if dataset_choice == "Kaggle" and preprocessor is not None:
            # X_single contains the 43 raw kaggle columns (ordered by EXPECTED)
            try:
                Xp = preprocessor.transform(X_single[EXPECTED])  # may return sparse
                if sparse.issparse(Xp):
                    X_input = Xp  # many models (xgboost) accept sparse; else conversion below
                else:
                    X_input = np.asarray(Xp)
            except Exception as e:
                st.error(f"Preprocessor.transform failed: {e}")
                raise e
        else:
            # Microsoft or no preprocessor: use DataFrame -> numeric array (apply scaler if present)
            X_tmp = force_to_expected(X_single, EXPECTED)
            X_input = prepare_array(X_tmp, scaler=scaler)

        # If model requires dense array and we have sparse, convert (small inputs okay)
        try:
            if sparse.issparse(X_input):
                # try predict with sparse first; if it errors, convert to dense and retry
                try:
                    if hasattr(m, "predict_proba"):
                        probs = get_positive_probs(m, X_input)
                        preds = m.predict(X_input)
                    else:
                        preds = m.predict(X_input)
                        probs = None
                except Exception:
                    X_input = X_input.toarray()
                    if hasattr(m, "predict_proba"):
                        probs = get_positive_probs(m, X_input)
                        preds = m.predict(X_input)
                    else:
                        preds = m.predict(X_input)
                        probs = None
            else:
                # X_input is dense numpy array
                if hasattr(m, "predict_proba"):
                    probs = get_positive_probs(m, X_input)
                    preds = m.predict(X_input)
                else:
                    preds = m.predict(X_input)
                    probs = None
        except Exception as e:
            st.error(f"Model predict failed: {e}")
            raise e

        # Show results
        if probs is not None:
            st.metric("Predicted probability (class=1)", f"{float(probs[0]):.4f}")
        st.write("Predicted label:", int(preds[0]))
        st.success("Prediction done.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")

# ---------- Batch evaluate ----------
st.header("Batch evaluate (upload processed CSV)")
uploaded_batch = st.file_uploader(f"Upload processed CSV for {dataset_choice} (multiple rows)", type=["csv"], key=f"batch_{dataset_choice}")
if uploaded_batch:
    raw = pd.read_csv(uploaded_batch)

    # preview / load-first-row buttons
    if not raw.empty:
        col_preview, col_load = st.columns([1,1])
        with col_preview:
            if st.button("Preview first row as single-record", key=f"preview_{dataset_choice}"):
                first = raw.iloc[[0]].copy()
                if dataset_choice == "Microsoft":
                    first = coerce_grade_column(first, models_dir)
                first_for_display = force_to_expected(first, EXPECTED)
                st.write("First row (forced to expected features):")
                st.dataframe(first_for_display.T)
        with col_load:
            if st.button("Load first row into Single-record form", key=f"load_{dataset_choice}"):
                first = raw.iloc[[0]].copy()
                if dataset_choice == "Microsoft":
                    first = coerce_grade_column(first, models_dir)
                first_for_display = force_to_expected(first, EXPECTED)
                st.session_state["loaded_single_row"] = first_for_display.iloc[0].to_dict()
                st.success("Loaded first row into Single-record form.")

    st.write("Uploaded preview (first 5 rows):")
    st.dataframe(raw.head())

    # target extraction
    if TARGET_COL in raw.columns:
        y_true = raw[TARGET_COL].values
        X_df = raw.drop(columns=[TARGET_COL])
    else:
        y_true = None
        X_df = raw.copy()
        st.warning("No target found — predictions only.")

    # coerce grade if Microsoft
    if dataset_choice == "Microsoft":
        X_df = coerce_grade_column(X_df, models_dir)

    X_for_eval = force_to_expected(X_df, EXPECTED)
    st.write("Final input shape:", X_for_eval.shape)

    # Note: for Kaggle keep the raw 43-col DF in X_for_eval; preprocessor will expand when needed
    arr = prepare_array(X_for_eval, scaler=scaler)

    eval_results = []
    preds_df = pd.DataFrame(index=X_for_eval.index)

    # model loop
    for mf in model_files:
        name = mf.stem
        try:
            mobj = joblib.load(mf)
        except Exception as e:
            st.warning(f"Skipping {name}: failed to load ({e})")
            continue

        # skip non-model objects (e.g., dicts, preprocessor saved as joblib)
        if not (hasattr(mobj, "predict") or hasattr(mobj, "predict_proba")):
            if debug_toggle:
                st.sidebar.write(f"Skipped {name}: no predict/predict_proba")
            continue

        # Prepare per-model input:
        input_arr = None
        # Try to detect what model expects:
        n_expected = getattr(mobj, "n_features_in_", None)
        feature_names_in = getattr(mobj, "feature_names_in_", None)
        # Try XGBoost booster names
        xgb_expected = None
        try:
            if hasattr(mobj, "get_booster"):
                booster = mobj.get_booster()
                xgb_expected = getattr(booster, "feature_names", None)
        except Exception:
            xgb_expected = None


        needs_expanded = False
        if (isinstance(n_expected, int) and n_expected > len(EXPECTED)) or (isinstance(xgb_expected, (list, tuple)) and len(xgb_expected) > len(EXPECTED)):
            needs_expanded = True

        try:
            if needs_expanded:
                # ensure preprocessor exists
                if preprocessor is None:
                    st.warning(f"Model {name} expects expanded features but no preprocessor found. Skipping.")
                    continue
                # transform using preprocessor on the raw EXPECTED columns
                try:
                    Xp = preprocessor.transform(X_for_eval[EXPECTED])
                    if sparse.issparse(Xp):
                        input_arr = Xp
                    else:
                        input_arr = np.asarray(Xp)
                except Exception as e:
                    st.warning(f"Preprocessor.transform failed for model {name}: {e}")
                    continue
            else:
                # model expects raw 41-ish features -> align by feature_names_in_ if available, else by EXPECTED
                if feature_names_in is not None:
                    exp_names = list(feature_names_in)
                    Xtmp = X_for_eval.copy()
                    # add missing -> 0
                    for c in exp_names:
                        if c not in Xtmp.columns:
                            Xtmp[c] = 0
                    # reorder and slice
                    Xtmp = Xtmp[exp_names].copy()
                    input_arr = Xtmp.values
                else:
                    # fallback: use EXPECTED order (will fill missing already due to force_to_expected earlier)
                    Xtmp = X_for_eval.copy()
                    for c in EXPECTED:
                        if c not in Xtmp.columns:
                            Xtmp[c] = 0
                    Xtmp = Xtmp[EXPECTED].copy()
                    input_arr = Xtmp.values
        except Exception as e:
            st.warning(f"Failed preparing input for {name}: {e}")
            continue

        # Convert sparse/dense as needed for predict
        try:
            if sparse.issparse(input_arr):
                # attempt predict with sparse first; convert to dense if model errors
                try:
                    if hasattr(mobj, "predict_proba"):
                        probs_all = mobj.predict_proba(input_arr)
                    else:
                        probs_all = None
                        y_pred = mobj.predict(input_arr)
                except Exception:
                    input_arr = input_arr.toarray()
                    if hasattr(mobj, "predict_proba"):
                        probs_all = mobj.predict_proba(input_arr)
                    else:
                        y_pred = mobj.predict(input_arr)
            else:
                if hasattr(mobj, "predict_proba"):
                    probs_all = mobj.predict_proba(input_arr)
                else:
                    probs_all = None
                    y_pred = mobj.predict(input_arr)
        except Exception as e:
            st.warning(f"Model {name} failed during predict: {e}")
            continue

        # extract positive-class probabilities robustly
        probs = None
        if probs_all is not None:
            classes = getattr(mobj, "classes_", None)
            if classes is not None and 1 in list(classes):
                pos_idx = list(classes).index(1)
            else:
                pos_idx = probs_all.shape[1] - 1
            probs = probs_all[:, pos_idx]
            y_pred = (probs >= 0.5).astype(int)

        # silent inversion if AUC < 0.5 (keeps your behavior)
        auc = np.nan
        if probs is not None and y_true is not None:
            try:
                auc = roc_auc_score(y_true, probs)
                if auc < 0.5:
                    probs = 1.0 - probs
                    y_pred = (probs >= 0.5).astype(int)
                    if debug_toggle:
                        st.sidebar.warning(f"{name}: AUC < 0.5 — inverted probabilities.")
            except Exception:
                auc = np.nan

        # compute metrics if target exists
        if y_true is not None:
            try:
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
            except Exception:
                acc = prec = rec = f1 = np.nan
        else:
            acc = prec = rec = f1 = np.nan

        eval_results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc})
        preds_df[f"{name}_pred"] = y_pred
        if probs is not None:
            preds_df[f"{name}_prob"] = probs

        if debug_toggle:
            st.sidebar.write(f"{name} input shape: {getattr(input_arr,'shape',None)}, model expects n_features_in_={getattr(mobj,'n_features_in_',None)}")

    # end model loop

    # display results
    if eval_results:
        res_df = pd.DataFrame(eval_results).sort_values(by="F1", ascending=False, na_position="last")
        st.write("Evaluation results:")
        st.dataframe(res_df)
    else:
        st.info("No model produced results.")

    # download predictions
    if not preds_df.empty:
        id_cols = [c for c in ["memberId","loanId","loan_amnt"] if c in raw.columns]
        out_df = pd.concat([raw[id_cols].reset_index(drop=True) if id_cols else raw.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)
        st.write("Predictions preview (first 5 rows):")
        st.dataframe(out_df.head())
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv_bytes, file_name=f"{dataset_choice}_predictions.csv", mime="text/csv")
    else:
        st.info("No predictions to download.")

st.markdown("---")
st.caption("Combined app — Microsoft & Kaggle. Use sidebar to switch dataset.")
