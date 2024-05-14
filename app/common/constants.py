from torch import device, cuda

TEMP_PATH = '/temp'
SAVED_MODELS_PATH = 'app/models/saved'
AUTOENCODER_PATH = SAVED_MODELS_PATH + '/autoencoder_dict.pt'
FINE_TUNED_AUTOENCODER_PATH = SAVED_MODELS_PATH + '/fine_tuned_autoencoder_dict.pt'
NEAREST_NEIGHBOR_PATH = SAVED_MODELS_PATH + '/nearest_neighbors.pkl'
DEVICE = device('cuda' if cuda.is_available() else 'cpu')
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_NEIGHBORS = 5
