FROM python:3.11.9-bullseye

# copy code
RUN mkdir /app
COPY requirements.txt /app
COPY app /app

# install requirements
RUN pip install -r /app/requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
