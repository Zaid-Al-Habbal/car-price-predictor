# 1. Base image
FROM python:3.13-slim

# 2. Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Set working directory
WORKDIR /app

# 4. Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy dependency definitions
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy application code
COPY app ./app
COPY models ./models

# 8. Expose port
EXPOSE 8000

# 9. Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
