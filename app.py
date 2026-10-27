from flask import Flask, render_template_string, request
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Dynamically locate decisionc.pkl in the same directory as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'decisionc.pkl')

model = None
model_load_error = None

try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"SUCCESS: Model loaded from {model_path}")
    else:
        model_load_error = f"Model file 'decisionc.pkl' was NOT found at: {model_path}"
        print(model_load_error)
except Exception as e:
    model_load_error = f"Error loading model: {e}"
    print(model_load_error)

# HTML Template embedded directly in Python
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Campus Placement Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8fafc;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .card {
            border: none;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .btn-primary {
            background-color: #2563eb;
            border: none;
            padding: 10px;
            font-weight: 600;
        }
        .btn-primary:hover {
            background-color: #1d4ed8;
        }
        .result-box {
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            font-size: 1.25rem;
            font-weight: bold;
            margin-top: 20px;
        }
        .placed {
            background-color: #d1fae5;
            color: #065f46;
            border: 1px solid #34d399;
        }
        .not-placed {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #f87171;
        }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="text-center mb-5">
            <h1 class="fw-bold text-primary">🎓 Campus Placement Predictor</h1>
            <p class="text-muted">Enter academic details to check placement eligibility</p>
        </div>

        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="card p-4 bg-white">
                    {% if model_error %}
                        <div class="alert alert-danger" role="alert">
                            <strong>Critical Error:</strong> {{ model_error }}<br><br>
                            👉 <strong>Solution:</strong> Copy your <code>decisionc.pkl</code> file and paste it directly into this folder: <br>
                            <code>{{ base_dir }}</code>
                        </div>
                    {% endif %}

                    <form method="POST">
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Gender</label>
                                <select name="gender" class="form-select">
                                    <option value="M">Male</option>
                                    <option value="F">Female</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Work Experience</label>
                                <select name="workex" class="form-select">
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                </select>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label fw-semibold">10th (%)</label>
                                <input type="number" step="0.1" name="ssc_p" class="form-control" value="65.0" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">10th Board</label>
                                <select name="ssc_b" class="form-select">
                                    <option value="Central">Central</option>
                                    <option value="Others">Others</option>
                                </select>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label fw-semibold">12th (%)</label>
                                <input type="number" step="0.1" name="hsc_p" class="form-control" value="65.0" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">12th Board</label>
                                <select name="hsc_b" class="form-select">
                                    <option value="Central">Central</option>
                                    <option value="Others">Others</option>
                                </select>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label fw-semibold">12th Specialization</label>
                                <select name="hsc_s" class="form-select">
                                    <option value="Commerce">Commerce</option>
                                    <option value="Science">Science</option>
                                    <option value="Arts">Arts</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Degree (%)</label>
                                <input type="number" step="0.1" name="degree_p" class="form-control" value="65.0" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Degree Field</label>
                                <select name="degree_t" class="form-select">
                                    <option value="Comm&Mgmt">Comm&Mgmt</option>
                                    <option value="Sci&Tech">Sci&Tech</option>
                                    <option value="Others">Others</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Employability Test (%)</label>
                                <input type="number" step="0.1" name="etest_p" class="form-control" value="60.0" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label fw-semibold">MBA Specialization</label>
                                <select name="specialisation" class="form-select">
                                    <option value="Mkt&HR">Mkt&HR</option>
                                    <option value="Mkt&Fin">Mkt&Fin</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">MBA (%)</label>
                                <input type="number" step="0.1" name="mba_p" class="form-control" value="60.0" required>
                            </div>

                            <div class="col-12 mt-4">
                                <button type="submit" class="btn btn-primary w-100" {% if model_error %}disabled{% endif %}>Predict Placement Status</button>
                            </div>
                        </div>
                    </form>

                    {% if prediction %}
                        <div class="result-box {% if prediction == 'Placed' %}placed{% else %}not-placed{% endif %}">
                            {% if prediction == 'Placed' %}
                                🎉 Student is Likely to be PLACED!
                            {% else %}
                                ⚠️ Student is NOT PLACED
                            {% endif %}
                            {% if probability %}
                                <div class="fs-6 mt-2 text-muted">Confidence: {{ probability }}</div>
                            {% endif %}
                        </div>
                    {% endif %}

                    {% if error %}
                        <div class="alert alert-danger mt-3" role="alert">
                            Error: {{ error }}
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == 'POST':
        if model is None:
            error = "Prediction model is not loaded because decisionc.pkl is missing."
            return render_template_string(HTML_TEMPLATE, prediction=prediction, probability=probability, error=error, model_error=model_load_error, base_dir=BASE_DIR)

        try:
            # Standard standard dataset label mapping for Campus Placement dataset
            gender_map = {'M': 1, 'F': 0}
            board_map = {'Central': 1, 'Others': 0}
            hsc_s_map = {'Commerce': 1, 'Science': 2, 'Arts': 0}
            degree_t_map = {'Comm&Mgmt': 0, 'Sci&Tech': 1, 'Others': 2}
            workex_map = {'Yes': 1, 'No': 0}
            specialisation_map = {'Mkt&Fin': 0, 'Mkt&HR': 1}

            # Map form options to exact numeric categories used during model training
            input_data = pd.DataFrame([{
                'gender': gender_map.get(request.form.get('gender'), 1),
                'ssc_p': float(request.form.get('ssc_p', 0)),
                'ssc_b': board_map.get(request.form.get('ssc_b'), 1),
                'hsc_p': float(request.form.get('hsc_p', 0)),
                'hsc_b': board_map.get(request.form.get('hsc_b'), 1),
                'hsc_s': hsc_s_map.get(request.form.get('hsc_s'), 1),
                'degree_p': float(request.form.get('degree_p', 0)),
                'degree_t': degree_t_map.get(request.form.get('degree_t'), 0),
                'workex': workex_map.get(request.form.get('workex'), 0),
                'etest_p': float(request.form.get('etest_p', 0)),
                'specialisation': specialisation_map.get(request.form.get('specialisation'), 0),
                'mba_p': float(request.form.get('mba_p', 0))
            }])

            # Run prediction
            pred = model.predict(input_data)[0]
            proba = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None

            # Handle prediction output format (numeric 1/0 or string 'Placed')
            prediction = "Placed" if (str(pred) in ['1', 'Placed', 'placed', 'True']) else "Not Placed"
            if proba is not None:
                probability = f"{np.max(proba) * 100:.2f}%"

        except Exception as err:
            error = str(err)

    return render_template_string(HTML_TEMPLATE, prediction=prediction, probability=probability, error=error, model_error=model_load_error, base_dir=BASE_DIR)

if __name__ == '__main__':
    app.run(debug=True)