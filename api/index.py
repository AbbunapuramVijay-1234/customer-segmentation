from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='../templates')

# Resolve paths correctly for Vercel
current_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(current_dir, 'model.pkl'))
scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))

SEGMENT_LABELS = {
    0: {"name": "Sensible / Budget", "desc": "Low income, low spending. Prefers essentials and budget offers."},
    1: {"name": "Careless / Spenders", "desc": "Low income, high spending. Highly responsive to sales and discounts."},
    2: {"name": "Careful / Savers", "desc": "High income, low spending. Needs premium value proposition to convert."},
    3: {"name": "Target / VIP", "desc": "High income, high spending. Prime candidates for luxury campaigns."}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        income = float(data.get('income'))
        spending_score = float(data.get('spending_score'))

        # Scale and predict
        scaled = scaler.transform([[income, spending_score]])
        cluster_id = int(model.predict(scaled)[0])

        result = SEGMENT_LABELS.get(cluster_id, {"name": "Unknown", "desc": "No details available."})

        return jsonify({
            "status": "success",
            "cluster": cluster_id,
            "segment_name": result["name"],
            "description": result["desc"]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)