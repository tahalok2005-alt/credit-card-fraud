"""
Flask app for Simple Credit Card Fraud Detection (FOML project)
Single-file app: upload CSV -> train & evaluate RandomForest -> show metrics, plots, explanations -> save model

Usage:
    1) pip install flask pandas numpy scikit-learn matplotlib joblib
    2) python app.py
    3) Visit http://127.0.0.1:5000/

Notes:
    - Expects the dataset to have a binary column named "Class" for fraud label (0 = non-fraud, 1 = fraud).
    - If dataset contains "Amount", the script will scale it into "Amount_scaled".
    - The app trains a RandomForest on the uploaded file (on-the-fly). For production, you may want to load a pre-trained model.
"""

import os
import io
import base64
import traceback
from datetime import datetime

from flask import Flask, request, render_template_string, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib

# ------------- Config -------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"csv"}
MODEL_FILENAME = "rf_fraud_model.joblib"

# Default model/hyperparams (can be tweaked in the UI form)
DEFAULTS = {
    "resample_undersample": True,
    "random_state": 42,
    "test_size": 0.2,
    "n_estimators": 150,
    "max_depth": 8,
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "change_this_secret_in_production"

# ---------- Utility functions ----------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def save_plot_to_base64(fig):
    """Return PNG image data in base64 for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64

def plot_roc(y_true, y_proba, auc):
    fig, ax = plt.subplots(figsize=(6,4))
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    ax.plot([0,1],[0,1], linestyle="--", linewidth=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return save_plot_to_base64(fig)

def plot_pr(y_true, y_proba, ap):
    fig, ax = plt.subplots(figsize=(6,4))
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax.plot(recall, precision, label=f"AP={ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    return save_plot_to_base64(fig)

def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    classes = ["Non-Fraud (0)", "Fraud (1)"]
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i,j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    return save_plot_to_base64(fig)

def plot_feature_importances(fi_series, top_n=10):
    top = fi_series.head(top_n)
    fig, ax = plt.subplots(figsize=(6, max(3, top_n*0.4)))
    top[::-1].plot(kind="barh", ax=ax)  # reverse for horizontal bar
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return save_plot_to_base64(fig)

# ---------- Core ML pipeline ----------

def run_pipeline(df,
                 target_col="Class",
                 resample_undersample=True,
                 random_state=42,
                 test_size=0.2,
                 n_estimators=150,
                 max_depth=8):
    """
    Runs preprocessing, optional undersampling, training, and evaluation.
    Returns a dictionary with metrics, images (base64), feature importances, and saved_model_path.
    Raises Exceptions with a helpful message when something is wrong.
    """
    results = {}
    # Basic checks
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not present in uploaded file. Please ensure the file has a binary column named '{target_col}' (0/1).")

    # Prepare X, y
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(int).copy()

    n_total = len(df)
    n_fraud = int(y.sum())
    results["dataset_rows"] = n_total
    results["fraud_count"] = n_fraud
    results["fraud_frac"] = n_fraud / n_total if n_total else 0.0

    # Scale 'Amount' if present
    if "Amount" in X.columns:
        scaler_amount = StandardScaler()
        X["Amount_scaled"] = scaler_amount.fit_transform(X[["Amount"]])
        X = X.drop(columns=["Amount"])
        results["scaled_amount"] = True
    else:
        results["scaled_amount"] = False

    # Optional undersampling (simple approach)
    if resample_undersample:
        df_all = pd.concat([X, y], axis=1)
        fraud = df_all[df_all[target_col] == 1]
        nonfraud = df_all[df_all[target_col] == 0]
        if len(fraud) == 0:
            raise ValueError("No positive (fraud) cases found in the dataset; cannot train. Please provide data with at least one fraud case.")
        # Keep a 4:1 nonfraud:fraud ratio (you can change)
        n_nonfraud_keep = max(len(fraud)*4, 1)
        nonfraud_down = resample(nonfraud,
                                 replace=False,
                                 n_samples=min(len(nonfraud), n_nonfraud_keep),
                                 random_state=random_state)
        df_balanced = pd.concat([fraud, nonfraud_down])
        df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
        X = df_balanced.drop(columns=[target_col])
        y = df_balanced[target_col].astype(int)
        results["after_undersample_shape"] = X.shape
        results["after_undersample_counts"] = y.value_counts().to_dict()
    else:
        results["after_undersample_shape"] = X.shape
        results["after_undersample_counts"] = y.value_counts().to_dict()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        stratify=y,
                                                        random_state=random_state)
    results["train_shape"] = X_train.shape
    results["test_shape"] = X_test.shape

    # Scale all features
    scaler_all = StandardScaler()
    X_train_scaled = scaler_all.fit_transform(X_train)
    X_test_scaled = scaler_all.transform(X_test)

    # Train RandomForest
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=random_state,
                                class_weight="balanced")
    rf.fit(X_train_scaled, y_train)

    # Predict & evaluate
    y_pred = rf.predict(X_test_scaled)
    if hasattr(rf, "predict_proba"):
        y_proba = rf.predict_proba(X_test_scaled)[:, 1]
    else:
        # fallback: use decision function
        y_proba = rf.decision_function(X_test_scaled)
        # scale to 0-1
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-9)

    auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    results["classification_report"] = classification_report(y_test, y_pred, digits=4, output_dict=True)
    results["roc_auc"] = float(auc)
    results["average_precision"] = float(avg_precision)

    cm = confusion_matrix(y_test, y_pred)
    results["confusion_matrix"] = cm.tolist()

    # Feature importances
    fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    results["feature_importances_top10"] = fi.head(10).to_dict()

    # Plots as base64
    results["plot_roc_b64"] = plot_roc(y_test, y_proba, auc)
    results["plot_pr_b64"] = plot_pr(y_test, y_proba, avg_precision)
    results["plot_confusion_b64"] = plot_confusion(cm)
    results["plot_fi_b64"] = plot_feature_importances(fi, top_n=min(10, len(fi)))

    # Save model + scaler together
    model_package = {
        "model": rf,
        "scaler": scaler_all,
        "trained_on": datetime.utcnow().isoformat() + "Z",
        "meta": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        }
    }
    joblib.dump(model_package, MODEL_FILENAME)
    results["saved_model"] = MODEL_FILENAME

    # Short summary
    results["summary"] = {
        "dataset_rows": int(results["dataset_rows"]),
        "fraud_count": int(results["fraud_count"]),
        "model": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "test_size": test_size,
        "roc_auc": float(auc),
        "average_precision": float(avg_precision),
    }

    return results

# ---------- Flask routes & templates ----------

INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>FOML - Credit Card Fraud Detection (Flask)</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Bootstrap CDN for quick styling -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { padding-top: 1.5rem; background:#f8f9fb; }
      .card { box-shadow: 0 4px 14px rgba(0,0,0,0.03); }
      .explain { font-size: 0.95rem; color:#444; }
      .small-mono { font-family: monospace; font-size: 0.85rem; background:#eef2ff; padding:0.25rem 0.5rem; border-radius:4px; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card p-4 mb-4">
        <h3>FOML — Credit Card Fraud Detection</h3>
        <p class="explain">Upload your CSV (must contain a binary <span class="small-mono">Class</span> column with 0/1 values). The app will train a RandomForest and show metrics and plots.</p>
        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            {% for cat, msg in messages %}
              <div class="alert alert-{{cat}}">{{msg}}</div>
            {% endfor %}
          {% endif %}
        {% endwith %}
        <form method="post" action="{{ url_for('upload') }}" enctype="multipart/form-data" class="row g-3">
          <div class="col-md-6">
            <label for="file" class="form-label">Choose CSV file</label>
            <input class="form-control" type="file" id="file" name="file" accept=".csv" required>
          </div>

          <div class="col-md-6">
            <label class="form-label">Options</label>
            <div class="form-check">
              <input class="form-check-input" type="checkbox" id="resample" name="resample" {% if defaults.resample_undersample %}checked{% endif %}>
              <label class="form-check-label" for="resample">Undersample majority class (simple balancing)</label>
            </div>
            <div class="row mt-2">
              <div class="col">
                <label class="form-label">Test size (0-1)</label>
                <input class="form-control" name="test_size" value="{{ defaults.test_size }}">
              </div>
              <div class="col">
                <label class="form-label">n_estimators</label>
                <input class="form-control" name="n_estimators" value="{{ defaults.n_estimators }}">
              </div>
              <div class="col">
                <label class="form-label">max_depth</label>
                <input class="form-control" name="max_depth" value="{{ defaults.max_depth }}">
              </div>
            </div>
          </div>

          <div class="col-12">
            <button class="btn btn-primary">Upload & Run</button>
            <a href="{{ url_for('download_model') }}" class="btn btn-outline-secondary">Download last saved model</a>
          </div>
        </form>
        <hr>
        <p class="mb-0"><small><strong>Tip:</strong> Use the original public 'creditcard.csv' (with anonymized V1..V28 features and Amount, Time columns) or any dataset with the same structure and a <span class="small-mono">Class</span> column.</small></p>
      </div>

      {% if results %}
      <div class="card p-4 mb-4">
        <h5>Results & Explanation</h5>
        <p class="explain">
          <strong>Summary:</strong> dataset rows: {{ results.summary.dataset_rows }}, fraud cases: {{ results.summary.fraud_count }}.
          The model trained is a RandomForest (n_estimators={{ results.summary.n_estimators }}, max_depth={{ results.summary.max_depth }}).
        </p>

        <div class="row">
          <div class="col-md-6">
            <h6>Metrics</h6>
            <table class="table table-sm table-bordered">
              <tr><th>ROC AUC</th><td>{{ results.summary.roc_auc | round(4) }}</td></tr>
              <tr><th>Average Precision (AP)</th><td>{{ results.summary.average_precision | round(4) }}</td></tr>
              <tr><th>Train shape</th><td>{{ results.train_shape }}</td></tr>
              <tr><th>Test shape</th><td>{{ results.test_shape }}</td></tr>
              <tr><th>Saved model</th><td><span class="small-mono">{{ results.saved_model }}</span></td></tr>
            </table>

            <h6>Explanation (quick)</h6>
            <p class="explain">
              <strong>ROC AUC</strong> measures the model's ability to rank fraud vs non-fraud; 1.0 is perfect, 0.5 is random.
              <br><strong>Average Precision</strong> (AP) summarizes the precision-recall curve and is useful for imbalanced datasets.
            </p>
          </div>

          <div class="col-md-6">
            <h6>Confusion Matrix</h6>
            <img src="data:image/png;base64,{{ results.plot_confusion_b64 }}" class="img-fluid" alt="confusion">
          </div>
        </div>

        <hr>

        <div class="row">
          <div class="col-md-6">
            <h6>ROC Curve</h6>
            <img src="data:image/png;base64,{{ results.plot_roc_b64 }}" class="img-fluid" alt="roc">
          </div>
          <div class="col-md-6">
            <h6>Precision - Recall</h6>
            <img src="data:image/png;base64,{{ results.plot_pr_b64 }}" class="img-fluid" alt="pr">
          </div>
        </div>

        <hr>
        <div class="row">
          <div class="col-md-6">
            <h6>Top Feature Importances</h6>
            <img src="data:image/png;base64,{{ results.plot_fi_b64 }}" class="img-fluid" alt="fi">
          </div>
          <div class="col-md-6">
            <h6>Classification Report</h6>
            <pre style="background:#f7f9fc; padding:10px; border-radius:6px;">{{ results.classification_text }}</pre>
          </div>
        </div>

        <hr>
        <h6>Notes & Recommendations</h6>
        <ul>
          <li>Undersampling is a quick way to handle class imbalance but may discard data. Consider <em>SMOTE</em> or class-weighted models for production.</li>
          <li>RandomForest gives feature importances; high importance features are worth investigating for business rules or alerts.</li>
          <li>For deployment, train offline on full data and load the saved model instead of retraining on each upload.</li>
        </ul>

      </div>
      {% endif %}

      <footer class="text-center small text-muted mb-4">FOML Flask app • Built for quick demos. Change hyperparameters with the form above.</footer>
    </div>
  </body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, defaults=DEFAULTS, results=None)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            flash("No file part", "danger")
            return redirect(url_for("index"))
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file", "danger")
            return redirect(url_for("index"))
        if file and allowed_file(file.filename):
            fname = secure_filename(file.filename)
            saved_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            file.save(saved_path)

            # parse options from form
            resample = request.form.get("resample") == "on"
            try:
                test_size = float(request.form.get("test_size", DEFAULTS["test_size"]))
            except:
                test_size = DEFAULTS["test_size"]
            try:
                n_estimators = int(request.form.get("n_estimators", DEFAULTS["n_estimators"]))
            except:
                n_estimators = DEFAULTS["n_estimators"]
            try:
                max_depth = int(request.form.get("max_depth", DEFAULTS["max_depth"]))
            except:
                max_depth = DEFAULTS["max_depth"]

            # Read CSV
            df = pd.read_csv(saved_path)
            # Run pipeline
            results = run_pipeline(df,
                                   target_col="Class",
                                   resample_undersample=resample,
                                   random_state=DEFAULTS["random_state"],
                                   test_size=test_size,
                                   n_estimators=n_estimators,
                                   max_depth=max_depth)

            # classification report text for display
            results["classification_text"] = pd.DataFrame(results["classification_report"]).transpose().to_string()

            flash("Pipeline completed successfully — model saved.", "success")
            return render_template_string(INDEX_HTML, defaults=DEFAULTS, results=results)
        else:
            flash("Only CSV files are allowed.", "danger")
            return redirect(url_for("index"))
    except Exception as e:
        tb = traceback.format_exc()
        flash(f"Error during processing: {e}", "danger")
        # For debugging: also show traceback in console and give user a short helpful message on page
        print(tb)
        return redirect(url_for("index"))

@app.route("/download-model", methods=["GET"])
def download_model():
    if os.path.exists(MODEL_FILENAME):
        return send_from_directory(".", MODEL_FILENAME, as_attachment=True)
    else:
        flash("No saved model yet. Upload & run to create a model.", "warning")
        return redirect(url_for("index"))

if __name__ == "__main__":
    # Run app
    app.run(debug=False, host="127.0.0.1", port=5000)
