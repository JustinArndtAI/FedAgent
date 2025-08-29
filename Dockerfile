FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download DistilBERT model during build
RUN python -c "from transformers import DistilBertTokenizer, DistilBertModel; \
    DistilBertTokenizer.from_pretrained('distilbert-base-uncased'); \
    DistilBertModel.from_pretrained('distilbert-base-uncased')"

# Copy all model and app files
COPY app.py .
COPY v6_final_*.pkl ./
COPY v6_final_*.json ./
COPY v4_*.pkl ./

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

# Run the application
CMD ["python", "app.py"]
