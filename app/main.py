from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from typing import Annotated
import numpy as np
from PIL import Image
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from app.models.state import State
from app.common.constants import DEFAULT_K_NEIGHBORS, SAVED_MODELS_PATH, IMAGES_DESC, IMAGE_DESC, \
    N_NEIGHBORS_DESC, MODEL_NOT_FOUND, OK, FIT_MODEL_DESC, PREDICT_SIMILARITY_DESC, DELETE_FITTED_MODELS_DESC, \
    MODEL_NAME_DESC
from app.common.tools import extract_archive_to_numpy, delete_if_exist, pil_image_to_numpy

app = FastAPI()
state = State().load_from_file()


@app.post('/fit_model', description=FIT_MODEL_DESC)
def fit_model(
        model_name: Annotated[str, Query(max_length=50, description=MODEL_NAME_DESC)],
        images: UploadFile = File(description=IMAGES_DESC)
):
    state.knn.update({model_name: NearestNeighbors(n_neighbors=DEFAULT_K_NEIGHBORS)})
    data = extract_archive_to_numpy(images)
    embeddings = state.cnn(data)
    state.knn[model_name].fit(embeddings)
    state.save_knn_model(model_name)
    return {
        'message': OK
    }


@app.post('/predict_similarity', description=PREDICT_SIMILARITY_DESC)
def predict_similarity(
        model_name: Annotated[str, Query(max_length=50, description=MODEL_NAME_DESC)],
        image: UploadFile = File(description=IMAGE_DESC),
        k_neighbors: Annotated[int, Query(description=N_NEIGHBORS_DESC)] = DEFAULT_K_NEIGHBORS
):
    knn = state.knn.get(model_name)
    if not knn:
        raise HTTPException(status_code=404, detail=MODEL_NOT_FOUND)

    k_neighbors = knn.n_samples_fit_ if k_neighbors > knn.n_samples_fit_ else k_neighbors

    data = pil_image_to_numpy(Image.open(image.file))
    data = np.expand_dims(data, axis=0)
    embedding = state.cnn(data)

    neighbors_ids = knn.kneighbors(embedding.reshape(1, -1), k_neighbors, return_distance=False)[0]
    neighbors_embeddings = knn._fit_X[neighbors_ids]

    probabilities = cosine_similarity(embedding, neighbors_embeddings)
    probabilities_percent = int(np.mean(probabilities) * 100)
    return {
        'message': f'{probabilities_percent}%'
    }


@app.delete('/delete_models', description=DELETE_FITTED_MODELS_DESC)
def clear_state(
        model_name: Annotated[str, Query(max_length=50, description=MODEL_NAME_DESC)],
):
    if model_name not in state.knn.keys():
        raise HTTPException(status_code=404, detail=MODEL_NOT_FOUND)

    delete_if_exist(f'{SAVED_MODELS_PATH + model_name}-knn.pkl')
    del state.knn[model_name]
    return {
        'message': OK
    }
