import io
import os
import shutil
import zipfile
import numpy as np
from PIL import Image
from fastapi import UploadFile

from app.common.constants import UNKNOWN_ARCHIVE, DEFAULT_HEIGHT, DEFAULT_WIDTH


def delete_if_exist(path, is_directory=False):
    if os.path.exists(path):
        if is_directory:
            shutil.rmtree(path)
        else:
            os.remove(path)


def pil_image_to_numpy(image):
    image = image.resize((DEFAULT_HEIGHT, DEFAULT_WIDTH))
    image = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255
    return image


def extract_archive_to_numpy(file: UploadFile):
    file_type = file.filename.split('.')[-1]
    if file_type == 'zip':
        results = []
        with zipfile.ZipFile(file=io.BytesIO(file.file.read())) as zf:
            for file in zf.namelist():
                with zf.open(file) as f:
                    img = pil_image_to_numpy(Image.open(f))
                    results.append(img)
        return np.array(results)

    raise KeyError(f'{UNKNOWN_ARCHIVE} {file_type}')
