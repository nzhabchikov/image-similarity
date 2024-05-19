from sklearn.neighbors import NearestNeighbors
from typing import Annotated
import torch
from PIL import Image
from torch import cosine_similarity
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from app.models.tools import get_embeddings, get_transformer
from app.models.state import State
from app.models.cnn_models import get_mobilenet_model
from app.common.constants import TEMP_PATH, DEFAULT_N_NEIGHBORS, DEFAULT_RESIZE_HEIGHT, DEFAULT_RESIZE_WIDTH, \
    SAVED_MODELS_PATH, ModelType
from app.common.tools import unpack_archive, del_if_exist

app = FastAPI()
state = State().load_from_file()


@app.post('/train_model')
def train_model(
        images: UploadFile = File(...),
        model_type: ModelType = Query(...),
        scale_image_height: Annotated[int, Query()] = DEFAULT_RESIZE_HEIGHT,
        scale_image_width: Annotated[int, Query()] = DEFAULT_RESIZE_WIDTH,
):
    if model_type.value == ModelType.lite.value:
        state.cnn_model = get_mobilenet_model()
    state.n_neighbors_model = NearestNeighbors(n_neighbors=DEFAULT_N_NEIGHBORS)
    state.image_height = scale_image_height
    state.image_width = scale_image_width

    unpack_archive(images, TEMP_PATH)
    dataset = ImageFolder(TEMP_PATH, transform=get_transformer(state.image_height, state.image_width))

    embeddings = get_embeddings(model=state.cnn_model, data=dataset)
    state.n_neighbors_model.fit(embeddings)

    state.save_n_neighbors_model(model_type.value, size=(state.image_height, state.image_width))
    del_if_exist(TEMP_PATH, is_directory=True)
    return {
        'message': 'OK'
    }


@app.post('/predict_similarity')
def predict_similarity(
        image: UploadFile = File(...),
        n_neighbors: Annotated[int, Query()] = DEFAULT_N_NEIGHBORS
):
    if not state.n_neighbors_model:
        raise HTTPException(status_code=404, detail="Not found fitted model, use /train_model handler before")

    transforms = get_transformer(state.image_height, state.image_width)
    n_neighbors_model = state.n_neighbors_model
    n_neighbors = n_neighbors_model.n_samples_fit_ if n_neighbors > n_neighbors_model.n_samples_fit_ else n_neighbors

    image_tensor = transforms(Image.open(image.file)).unsqueeze(0)
    embedding = get_embeddings(model=state.cnn_model, data=TensorDataset(image_tensor, torch.zeros(1)))

    neighbors_ids = n_neighbors_model.kneighbors(embedding.reshape(1, -1), n_neighbors, return_distance=False)[0]
    neighbors_embeddings = n_neighbors_model._fit_X[neighbors_ids]

    probabilities = cosine_similarity(embedding, torch.from_numpy(neighbors_embeddings))
    probabilities_percent = int(torch.mean(probabilities) * 100)
    return {
        'message': f'{probabilities_percent}%'
    }


@app.delete('/clear_state')
def clear_state():
    del_if_exist(SAVED_MODELS_PATH, is_directory=True)
    state.clear_state()
    return {
        'message': 'OK'
    }
