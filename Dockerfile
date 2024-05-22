FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# copy code
RUN mkdir /workspace/app
COPY requirements.txt /workspace/app
COPY requirements-cuda121.txt /workspace/app
COPY app /workspace/app

# install requirements
RUN pip install -r /workspace/app/requirements.txt
RUN pip install -r /workspace/app/requirements-cuda121.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
