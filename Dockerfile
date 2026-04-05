# Default HF Space / CI entrypoint (build context = repo root).
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml setup.py README.md openenv.yaml ./
COPY agentguard_gym ./agentguard_gym
COPY server ./server

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
