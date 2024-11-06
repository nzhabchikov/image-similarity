# Image similarity
Сервис сравнивает изображение на предмет схожести с предварительно загруженной базой изображений.

# Запуск
### Docker
1. `docker build -t image-similarity .`
2. `docker run -p 8000:80 image-similarity`

### Windows/Linux/macOS
Для запуска необходима версия python 3.10+
1. `pip install -r requirements.txt`
2. `uvicorn app.main:app --port 8000 --host 0.0.0.0`


После запуска swagger документация доступна по адресу `http://localhost:8000/docs`.