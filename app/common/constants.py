from enum import Enum

from torch import device, cuda

TEMP_PATH = '/temp'
SAVED_MODELS_PATH = 'app/models/saved'
NEAREST_NEIGHBORS = 'nearest_neighbors'
DEVICE = device('cuda' if cuda.is_available() else 'cpu')
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_NEIGHBORS = 5
DEFAULT_RESIZE_HEIGHT, DEFAULT_RESIZE_WIDTH = 192, 192


class ModelType(Enum):
    standard = 'standard'
    lite = 'lite'
