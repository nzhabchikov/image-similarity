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
from app.models.cnn_models import get_cnn_model
from app.common.constants import TEMP_PATH, DEFAULT_N_NEIGHBORS, DEFAULT_RESIZE_HEIGHT, DEFAULT_RESIZE_WIDTH, \
    SAVED_MODELS_PATH, ModelType, SCALE_HEIGHT_DESC, SCALE_WIDTH_DESC, MODEL_TYPE_DESC, IMAGES_DESC, IMAGE_DESC, \
    N_NEIGHBORS_DESC, NOT_FOUND_FITTED_MODEL, OK, FIT_MODEL_DESC, PREDICT_SIMILARITY_DESC, DELETE_FITTED_MODELS_DESC
from app.common.tools import unpack_archive, del_if_exist

app = FastAPI()
state = State().load_from_file()


@app.post('/fit_model', description=FIT_MODEL_DESC)
def fit_model(
        images: UploadFile = File(description=IMAGES_DESC),
        model_type: ModelType = Query(description=MODEL_TYPE_DESC),
        scale_image_height: Annotated[int, Query(description=SCALE_HEIGHT_DESC)] = DEFAULT_RESIZE_HEIGHT,
        scale_image_width: Annotated[int, Query(description=SCALE_WIDTH_DESC)] = DEFAULT_RESIZE_WIDTH,
):
    state.cnn_model = get_cnn_model(model_type)
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
        'message': OK
    }


@app.post('/predict_similarity', description=PREDICT_SIMILARITY_DESC)
def predict_similarity(
        image: UploadFile = File(description=IMAGE_DESC),
        n_neighbors: Annotated[int, Query(description=N_NEIGHBORS_DESC)] = DEFAULT_N_NEIGHBORS
):
    if not state.n_neighbors_model:
        raise HTTPException(status_code=404, detail=NOT_FOUND_FITTED_MODEL)

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


@app.delete('/delete_fitted_models', description=DELETE_FITTED_MODELS_DESC)
def clear_state():
    del_if_exist(SAVED_MODELS_PATH, is_directory=True)
    state.clear_state()
    return {
        'message': OK
    }
