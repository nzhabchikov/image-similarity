import torch
import numpy as np
import os
import pickle
from sklearn.neighbors import NearestNeighbors
from app.models.autoencoder import AutoencoderModel
from app.common.constants import DEVICE, AUTOENCODER_PATH, FINE_TUNED_AUTOENCODER_PATH, NEAREST_NEIGHBOR_PATH, \
    DEFAULT_N_NEIGHBORS


def get_autoencoder_model():
    path = FINE_TUNED_AUTOENCODER_PATH if os.path.exists(FINE_TUNED_AUTOENCODER_PATH) else AUTOENCODER_PATH
    model = AutoencoderModel()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def get_nearest_neighbors_model():
    if os.path.exists(NEAREST_NEIGHBOR_PATH):
        model = pickle.load(open(NEAREST_NEIGHBOR_PATH, 'rb'))
        return model
    else:
        return NearestNeighbors(n_neighbors=DEFAULT_N_NEIGHBORS)


def save_autoencoder_model(model):
    model.eval()
    torch.save(model.state_dict(), FINE_TUNED_AUTOENCODER_PATH)


def save_nearest_neighbors_model(model):
    path = NEAREST_NEIGHBOR_PATH
    pickle.dump(model, open(path, 'wb'))


def compare_embeddings(target, neighbors):
    return [np.dot(target, neighbor) / np.linalg.norm(target) / np.linalg.norm(neighbor) for neighbor in neighbors]
