FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml setup.py README.md openenv.yaml ./
COPY aml_defense_gym ./aml_defense_gym
COPY server ./server

# Install only real PyPI-available dependencies (openenv-core is not on PyPI)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "fastapi>=0.104.0" \
        "uvicorn[standard]>=0.24.0" \
        "pydantic>=2.5.0" \
        "pandas>=2.0.0" \
        "openai>=1.40.0" \
    && pip install --no-cache-dir -e . --no-deps

# HF Spaces requires port 7860
EXPOSE 7860
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=5)"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
