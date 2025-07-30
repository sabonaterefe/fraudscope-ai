# ✅ Base image
FROM python:3.10-slim

# 📁 Set working directory
WORKDIR /app

# 🚀 Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 📦 Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 📂 Copy app files
COPY dashboard/ ./dashboard/
COPY data/processed/ ./data/processed/
COPY artifacts/ ./artifacts/

# 🌐 Set default command
CMD ["streamlit", "run", "dashboard/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
