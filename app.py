import math
import warnings
import sys
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings("ignore")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__)

data = joblib.load("loan_scoring_model.pkl")
model = data["model"]
scaler = data["scaler"]
le_employment = data["label_encoder_employment"]
le_decision = data["label_encoder_decision"]

# Store as plain Python str to avoid np.str_ comparison issues
EMPLOYMENT_TYPES = [str(c) for c in le_employment.classes_.tolist()]


# Observed P(approved) bounds from grid search over the model's feature space
_P_MIN = 0.03
_P_MAX = 0.62

_DECISIONS = {
    0: ("approved",     "Зөвшөөрөх",  "Таны зээлийн хүсэлт батлагдлаа."),
    1: ("manual_review","Гар шалгалт", "Таны хүсэлтийг нэмэлт шалгалтад илгээлээ."),
    2: ("rejected",     "Татгалзах",   "Таны зээлийн хүсэлт татгалзагдлаа."),
}


def compute_score(monthly_income, employment_years, requested_amount, employment_idx):
    emp_encoded = int(employment_idx)
    ratio = requested_amount / monthly_income
    dti = (requested_amount / 12) / monthly_income
    log_inc = math.log1p(monthly_income)
    log_amt = math.log1p(requested_amount)

    features = np.array([[monthly_income, employment_years, requested_amount,
                          ratio, dti, log_inc, log_amt, emp_encoded]])
    scaled = scaler.transform(features)
    proba = model.predict_proba(scaled)[0]
    pred_class = int(model.predict(scaled)[0])

    # Normalise P(approved) from observed [_P_MIN, _P_MAX] to [0, 1000]
    p_approved = float(proba[0])
    score = int(round(max(0, min(1000,
        (p_approved - _P_MIN) / (_P_MAX - _P_MIN) * 1000
    ))))

    decision_key, decision_label, decision_msg = _DECISIONS[pred_class]
    return score, proba.tolist(), decision_key, decision_label, decision_msg


@app.route("/")
def index():
    return render_template("index.html", employment_types=EMPLOYMENT_TYPES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        monthly_income = float(request.form["monthly_income"])
        employment_years = float(request.form["employment_years"])
        requested_amount = float(request.form["requested_amount"])
        employment_idx = int(request.form["employment_type"])

        if monthly_income <= 0 or requested_amount <= 0 or employment_years < 0:
            return jsonify({"error": "Утгууд тэгээс их байх ёстой."}), 400
        if employment_idx < 0 or employment_idx >= len(EMPLOYMENT_TYPES):
            return jsonify({"error": "Ажил эрхлэлтийн утга буруу."}), 400

        score, proba, decision_key, decision_label, decision_msg = compute_score(
            monthly_income, employment_years, requested_amount, employment_idx
        )

        return jsonify({
            "score": score,
            "decision": decision_key,
            "decision_label": decision_label,
            "decision_msg": decision_msg,
            "proba_approved": round(proba[0] * 100, 1),
            "proba_review": round(proba[1] * 100, 1),
            "proba_rejected": round(proba[2] * 100, 1),
        })
    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Оролтын алдаа: {e}"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
