from typing import Annotated
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset
import numpy as np
from fastapi import FastAPI, Query, UploadFile, File
from app.models.autoencoder import get_embeddings, get_transformer, fine_tune
from app.models.tools import get_autoencoder_model, get_nearest_neighbors_model, compare_embeddings, \
    save_autoencoder_model, save_nearest_neighbors_model
from app.common.constants import TEMP_PATH, DEFAULT_N_NEIGHBORS, FINE_TUNED_AUTOENCODER_PATH, NEAREST_NEIGHBOR_PATH
from app.common.tools import unpack_archive, del_if_exist

app = FastAPI()

model = get_autoencoder_model()
n_neighbors_clf = get_nearest_neighbors_model()


@app.post('/train_model')
def train_model(
        images: UploadFile = File(...)
):
    unpack_archive(images, TEMP_PATH)

    dataset = ImageFolder(TEMP_PATH, transform=get_transformer())
    fine_tune(model=model, train_data=dataset)

    embeddings = get_embeddings(model=model, data=dataset)
    n_neighbors_clf.fit(embeddings)

    save_autoencoder_model(model)
    save_nearest_neighbors_model(n_neighbors_clf)

    del_if_exist(TEMP_PATH, is_directory=True)
    return {
        'message': 'OK'
    }


@app.post('/predict_similarity')
def predict_similarity(
        image: UploadFile = File(...),
        n_neighbors: Annotated[int, Query()] = DEFAULT_N_NEIGHBORS
):
    transforms = get_transformer()

    image_tensor = transforms(Image.open(image.file))
    embedding = get_embeddings(model=model, data=TensorDataset(image_tensor.unsqueeze(0), torch.tensor([0])))

    n_neighbors = n_neighbors_clf.n_samples_fit_ if n_neighbors > n_neighbors_clf.n_samples_fit_ else n_neighbors
    neighbors_ids = n_neighbors_clf.kneighbors(embedding.reshape(1, -1),
                                               n_neighbors=n_neighbors, return_distance=False)[0]
    neighbors_embeddings = n_neighbors_clf._fit_X[neighbors_ids]

    probabilities = compare_embeddings(embedding, neighbors_embeddings)
    probabilities_percent = int(np.mean(probabilities) * 100)
    return {
        'message': f'{probabilities_percent if probabilities_percent > 0 else 0}%'
    }


@app.delete('/delete_trained_models')
def delete_trained_models():
    del_if_exist(FINE_TUNED_AUTOENCODER_PATH)
    del_if_exist(NEAREST_NEIGHBOR_PATH)
    return {
        'message': 'OK'
    }
