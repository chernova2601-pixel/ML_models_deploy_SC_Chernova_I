import logging
from flask import Flask, request, jsonify
from pythonjsonlogger import jsonlogger
from model_handler import ModelHandler

app = Flask(__name__)

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

model_handler = ModelHandler(model_path='models/model_v1.pkl')

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
    pred, prob = model_handler.predict(data['features'])
    response = {'prediction': pred, 'probability': prob}
    logger.info({"endpoint": "/predict", "features": data['features'], "response": response})
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
