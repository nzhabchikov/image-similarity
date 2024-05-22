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


NOT_FOUND_FITTED_MODEL = 'Not found fitted model, use /fit_model handler before.'
OK = 'OK'
UNKNOWN_ARCHIVE = 'Unknown archive type:'

# ru descriptions
MODEL_TYPE_DESC = (
    'Задает модель используемую для получения эмбеддингов. standard - MNASNet1_3, '
    'lite - MobileNet_V3_Large. MobileNet_V3_Large более быстрая модель по сравнению с MNASNet1_3, '
    'но немного хуже в способности отдалить эмбеддинги разных классов друг от друга.'
)
SCALE_HEIGHT_DESC = 'Высота в пикселях до которой будет сжато/растянуто изображение.'
SCALE_WIDTH_DESC = 'Ширина в пикселях до которой будет сжато/растянуто изображение.'
IMAGES_DESC = 'zip архив изображений.'
IMAGE_DESC = 'Изображение для сравнения.'
N_NEIGHBORS_DESC = 'Количество ближайших эмбеддингов участвующих в сравнении.'
FIT_MODEL_DESC = 'Получение эмбедингов изображений и обучение NearestNeighbors модели.'
PREDICT_SIMILARITY_DESC = 'Получение эмбеддинга изображения и сравнение с ближайшими эмбеддингами.'
DELETE_FITTED_MODELS_DESC = 'Удаление обученных моделей. Сброс состояния сервиса до изначального.'
