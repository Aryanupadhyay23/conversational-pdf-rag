FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the single port HF Spaces expects (7860)
EXPOSE 7860

# Streamlit configuration — must use port 7860 for HF Spaces
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Start FastAPI on internal port 8000, then Streamlit on 7860
CMD ["sh", "-c", "uvicorn api:app --host 127.0.0.1 --port 8000 & streamlit run app.py --server.port 7860 --server.address 0.0.0.0 --server.enableXsrfProtection false --server.enableCORS false"]