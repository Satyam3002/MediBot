FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .


RUN mkdir -p backend/saved_skin_model backend/saved_xray_model


EXPOSE 8000


CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port 8000 & python bot/bot_main.py"] 