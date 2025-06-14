FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV TORCH_HOME=/app/cache
ENV PYTHONPATH=/app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory
RUN mkdir -p /app/cache

# Copy the rest of the application
COPY . .

# Create directories for models if they don't exist
RUN mkdir -p backend/saved_skin_model backend/saved_xray_model

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application with optimized settings
CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port 8000 --workers 1 --limit-concurrency 10 & python bot/bot_main.py"] 