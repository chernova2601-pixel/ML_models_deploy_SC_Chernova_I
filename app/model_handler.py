import pickle
import numpy as np

class ModelHandler:
    def __init__(self, model_path='models/model_v1.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, features):
        features = np.array(features).reshape(1, -1)
        pred = self.model.predict(features)[0]
        prob = self.model.predict_proba(features)[0][1]
        return int(pred), float(prob)
