FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p database uploads face_encodings logs

ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

EXPOSE $PORT

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 app:app

