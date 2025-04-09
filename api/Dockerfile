FROM python:3.10-slim
LABEL author="Christophe Pimm"

# No prompt for user input
ENV DEBIAN_FRONTEND=noninteractive
#Python unbuffered output
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH="models"
ENV MODEL_FILE="unet_v11_multiclass_epoch_19.h5"

WORKDIR /app

RUN apt-get update && apt-get install -y curl
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY models /app/models
COPY damage_pred_api.py /app/damage_pred_api.py
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8080

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "damage_pred_api:app"]
