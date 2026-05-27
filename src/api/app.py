import pickle
import numpy as np
import logging
from pythonjsonlogger import jsonlogger
from flask import Flask, request, jsonify

app = Flask(__name__)

# Настройка логирования в JSON
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Загрузка модели
with open('models/credit_default_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/health', methods=['GET'])
def health():
    logger.info({"endpoint": "/health", "status": "ok"})
    return jsonify({'status': 'ok'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'features' not in data:
        logger.warning({"endpoint": "/predict", "error": "Missing features key"})
        return jsonify({'error': 'Missing "features" key'}), 400
    features = np.array(data['features']).reshape(1, -1)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    response = {'prediction': int(pred), 'probability': float(prob)}
    logger.info({"endpoint": "/predict", "features": data['features'], "response": response})
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
