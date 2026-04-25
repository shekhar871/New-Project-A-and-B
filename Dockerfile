# Default HF Space / CI entrypoint (build context = repo root).
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml setup.py README.md openenv.yaml ./
COPY agentguard_gym ./agentguard_gym
COPY server ./server
COPY data ./data

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

ENV PORT=7860
EXPOSE 7860
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','7860'), timeout=3)"

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
