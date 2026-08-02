FROM python:3.8-slim

LABEL app="banksys_sy_wangzhihao"

WORKDIR /app

# Install Python dependencies
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 -i "${PIP_INDEX_URL}" -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY app.py .

# Create writable directory for models
RUN mkdir -p /app/models && chmod 777 /app/models

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8888')" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8888", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
