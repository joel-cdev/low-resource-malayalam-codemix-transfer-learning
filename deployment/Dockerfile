# Base image — slim Python 3.10
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install gcc (needed to compile some Python packages)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
# Docker caches this layer — if requirements.txt hasn't changed, it won't reinstall
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .

# Copy the trained model folder into the container
# The model/ folder must exist in the same directory as this Dockerfile
COPY model/ ./model/

# Tell the app where to find the model
ENV MODEL_PATH=./model

# Expose port 8000
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
