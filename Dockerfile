FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    praat \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV VOICE_TASK=clean_detection

EXPOSE 7860

CMD ["python", "app.py"]