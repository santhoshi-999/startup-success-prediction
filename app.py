from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

app = Flask(__name__)

# Load trained model
model = joblib.load("random_forest_model.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predictor")
def predictor():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    os.makedirs("static", exist_ok=True)

    input_data = [
        float(request.form.get("f1")),
        float(request.form.get("f2")),
        float(request.form.get("f3")),
        float(request.form.get("f4")),
        float(request.form.get("f5")),
        float(request.form.get("f6")),
        float(request.form.get("f7")),
        float(request.form.get("f8")),
        float(request.form.get("f9"))
    ]

    feature_names = [
        'age_first_funding_year',
        'age_last_funding_year',
        'age_first_milestone_year',
        'age_last_milestone_year',
        'relationships',
        'funding_rounds',
        'funding_total_usd',
        'milestones',
        'avg_participants'
    ]

    df = pd.DataFrame([input_data], columns=feature_names)

    # Prediction
    prediction = model.predict(df)[0]

    probs = model.predict_proba(df)[0]
    class_index = list(model.classes_).index('acquired')
    probability = probs[class_index] * 100

    if prediction == 'acquired':
        result = "Acquired (Startup Success)"
    else:
        result = "Closed (Startup Failure)"

    # =============================
    # 1️⃣ Input Feature Graph
    # =============================
    plt.figure(figsize=(8,4))
    plt.bar(feature_names, input_data)
    plt.xticks(rotation=45)
    plt.title("Input Feature Values")
    plt.tight_layout()
    plt.savefig("static/input_features.png")
    plt.close()

    # =============================
    # 2️⃣ SHAP Graph
    # =============================
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    plt.figure()
    shap.summary_plot(shap_values, df, show=False)
    plt.tight_layout()
    plt.savefig("static/shap_dynamic.png")
    plt.close()

    # =============================
    # 3️⃣ Confusion Matrix (Static)
    # =============================
    from sklearn.metrics import confusion_matrix

    # dummy matrix for display (optional real one)
    cm = [[175, 12], [51, 39]]

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    return render_template(
        "result.html",
        prediction=result,
        probability=round(probability, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
