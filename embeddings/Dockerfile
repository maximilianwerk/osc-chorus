FROM python:3.10

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

# Download models from the internet and store in Docker image
RUN python code/app/minilm/loadModel.py

# Commenting out as there is no provision to save model in https://github.com/openai/CLIP
#RUN python code/app/clip/loadModel.py

# Download finetuned models from Jina
RUN python code/app/finetuned/loadModel.py

WORKDIR /code/app


CMD ["gunicorn", "main:app", "-b", "0.0.0.0:8000", "--timeout", "600", "-w", "1", "-k", "uvicorn.workers.UvicornWorker"]
EXPOSE 8000
