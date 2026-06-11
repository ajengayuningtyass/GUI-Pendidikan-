import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# ── Load model ──────────────────────────────────────────────────
with open('model_prestasi.pkl', 'rb') as f:
    artefak = pickle.load(f)

rf           = artefak['rf']
dt           = artefak['dt']
knn          = artefak['knn']
scaler       = artefak['scaler']
le_target    = artefak['le_target']
FEATURE_NAMES = artefak['feature_names']
CLASSES       = artefak['classes']   # ['Rendah', 'Sedang', 'Tinggi']

print("✅ Model loaded:", FEATURE_NAMES)
print("✅ Kelas:", CLASSES)


def encode_input(data: dict) -> np.ndarray:
    """
    Konversi dict input dari frontend ke array numerik sesuai urutan X.columns.
    Sesuaikan mapping di sini jika nama kolom dataset kamu berbeda.
    """
    gender_map  = {'Male': 1, 'Female': 0}
    dept_map    = {'Business': 0, 'Computer Science': 1,
                   'Electrical Engineering': 2, 'Mathematics': 3, 'Mechanical': 4}
    bool_map    = {'Yes': 1, 'No': 0}
    parent_map  = {'Employed': 0, 'Self-employed': 1, 'Unemployed': 2}
    income_map  = {'High': 0, 'Low': 1, 'Medium': 2}

    # Nilai default untuk fitur yang tidak ada di form (diisi median/0)
    feature_map = {col: 0 for col in FEATURE_NAMES}

    # Isi dari input form
    feature_map['Gender']                          = gender_map.get(data.get('gender', 'Male'), 0)
    feature_map['Age']                             = int(data.get('age', 20))
    feature_map['Department']                      = dept_map.get(data.get('dept', 'Computer Science'), 1)
    feature_map['Study_Hours_Per_Day']             = float(data.get('study', 5))
    feature_map['Attendance_Rate']                 = max(0, 100 - int(data.get('absent', 5)) * 3)
    feature_map['Internet_Access_at_Home']         = bool_map.get(data.get('internet', 'Yes'), 1)
    feature_map['Extracurricular_Activities']      = bool_map.get(data.get('activity', 'Yes'), 1)
    feature_map['Parent_Employment_Status']        = parent_map.get(data.get('parent', 'Employed'), 0)
    feature_map['Family_Income_Level']             = income_map.get(data.get('income', 'Medium'), 2)

    row = np.array([feature_map[col] for col in FEATURE_NAMES], dtype=float)
    return row


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        row      = encode_input(data).reshape(1, -1)
        row_sc   = scaler.transform(row)

        # Prediksi + probabilitas
        def run(model, X_input, use_scaled=False):
            inp = row_sc if use_scaled else row
            pred_enc = model.predict(inp)[0]
            pred_label = le_target.inverse_transform([pred_enc])[0]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(inp)[0]
                conf  = float(np.max(proba))
            else:
                conf = 1.0
            return pred_label, round(conf * 100, 1)

        rf_pred,  rf_conf  = run(rf,  row)
        dt_pred,  dt_conf  = run(dt,  row)
        knn_pred, knn_conf = run(knn, row, use_scaled=True)

        # Majority vote
        votes = [rf_pred, dt_pred, knn_pred]
        majority = max(set(votes), key=votes.count)

        return jsonify({
            'rf':       {'pred': rf_pred,  'conf': rf_conf},
            'dt':       {'pred': dt_pred,  'conf': dt_conf},
            'knn':      {'pred': knn_pred, 'conf': knn_conf},
            'majority': majority,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'classes': CLASSES})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
