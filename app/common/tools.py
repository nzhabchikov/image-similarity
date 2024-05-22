import io
import os
import shutil
import zipfile
from fastapi import UploadFile

from app.common.constants import UNKNOWN_ARCHIVE


def del_if_exist(path, is_directory=False):
    if os.path.exists(path):
        if is_directory:
            shutil.rmtree(path)
        else:
            os.remove(path)


def unpack_archive(file: UploadFile, path):
    file_type = file.filename.split('.')[-1]
    if file_type == 'zip':
        del_if_exist(path, is_directory=True)
        os.makedirs(path)
        with zipfile.ZipFile(file=io.BytesIO(file.file.read())) as zip_ref:
            zip_ref.extractall(path + path)
        return

    raise KeyError(f'{UNKNOWN_ARCHIVE} {file_type}')
