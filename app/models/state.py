import os
import pickle

from app.common.constants import NEAREST_NEIGHBORS, DEFAULT_RESIZE_HEIGHT, DEFAULT_RESIZE_WIDTH, \
    SAVED_MODELS_PATH, ModelType
from app.models.cnn_models import get_mobilenet_model, get_mnasnet_model
from app.common.tools import del_if_exist


class State:
    def __init__(self, ):
        self.n_neighbors_model = None
        self.cnn_model = None
        self.image_height = None
        self.image_width = None

    def load_default(self):
        self.cnn_model = get_mnasnet_model()
        self.image_height, self.image_width = DEFAULT_RESIZE_HEIGHT, DEFAULT_RESIZE_WIDTH
        return self

    def load_from_file(self):
        if os.path.exists(SAVED_MODELS_PATH):
            files = os.listdir(SAVED_MODELS_PATH)
            model_name = None
            for file_name in files:
                if file_name.startswith(NEAREST_NEIGHBORS):
                    model_name = file_name

            if model_name:
                self.n_neighbors_model = pickle.load(open(f'{SAVED_MODELS_PATH}/{model_name}', 'rb'))
                cnn_model_type, size = model_name.split('-')[1:]
                self.cnn_model = get_mobilenet_model() if cnn_model_type == ModelType.lite.value else get_mnasnet_model()
                self.image_height, self.image_width = size.split('.')[0].split('x')
                return self

        self.load_default()
        return self

    def save_n_neighbors_model(self, model_type, size):
        del_if_exist(SAVED_MODELS_PATH, is_directory=True)
        os.makedirs(SAVED_MODELS_PATH)
        path = f'{SAVED_MODELS_PATH}/{NEAREST_NEIGHBORS}-{model_type}-{size[0]}x{size[1]}.pkl'
        pickle.dump(self.n_neighbors_model, open(path, 'wb'))

    def clear_state(self):
        self.n_neighbors_model = None
        self.load_default()
