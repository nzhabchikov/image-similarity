SAVED_MODELS_PATH = 'app/models/saved/'
KNN = 'knn.pkl'
MNASNET = 'app/models/mnasnet.onnx'
DEFAULT_BATCH_SIZE = 32
DEFAULT_K_NEIGHBORS = 5
DEFAULT_HEIGHT, DEFAULT_WIDTH = 224, 224

MODEL_NOT_FOUND = 'Model not found.'
OK = 'OK'
UNKNOWN_ARCHIVE = 'Unknown archive type:'

# ru descriptions
IMAGES_DESC = 'zip архив изображений.'
IMAGE_DESC = 'Изображение для сравнения.'
N_NEIGHBORS_DESC = 'Количество ближайших эмбеддингов участвующих в сравнении.'
FIT_MODEL_DESC = 'Получение эмбедингов изображений и обучение NearestNeighbors модели.'
PREDICT_SIMILARITY_DESC = 'Получение эмбеддинга изображения и сравнение с ближайшими эмбеддингами.'
DELETE_FITTED_MODELS_DESC = 'Удаление модели.'
MODEL_NAME_DESC = 'Название модели.'
