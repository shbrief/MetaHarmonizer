FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_sm.txt ./
RUN pip install --no-cache-dir -r requirements_sm.txt

COPY src/ ./src/
COPY demo_nb/ ./demo_nb/
COPY readme.md ./readme.md
# COPY app_sm.py ./app_sm.py

ENV PYTHONPATH=/app
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app_sm.py", "--server.port=8501", "--server.address=0.0.0.0"]
