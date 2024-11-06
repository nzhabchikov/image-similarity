import os
import pickle

from app.common.constants import KNN, SAVED_MODELS_PATH
from app.models.onnx_model import ONNXModel


class State:
    def __init__(self):
        self.knn = {}
        self.cnn = ONNXModel()

    def load_from_file(self):
        if os.path.exists(SAVED_MODELS_PATH):
            files = os.listdir(SAVED_MODELS_PATH)
            if files:
                self.knn = {file.split('-knn.pkl')[0]: pickle.load(open(SAVED_MODELS_PATH + file, 'rb')) for file in
                            files}
        return self

    def save_knn_model(self, model_name):
        if not os.path.exists(SAVED_MODELS_PATH):
            os.makedirs(SAVED_MODELS_PATH)
        path = f'{SAVED_MODELS_PATH + model_name}-{KNN}'
        pickle.dump(self.knn[model_name], open(path, 'wb'))
