# A monitoring sidecar, not a training image: no torch, no sklearn, no jupyter.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn>=0.27"

COPY driftwatch ./driftwatch
COPY pyproject.toml README.md LICENSE ./

# run as a non-root user: this container is reachable from the scoring service
RUN useradd --create-home --uid 10001 driftwatch \
    && chown -R driftwatch:driftwatch /app
USER driftwatch

EXPOSE 8000

# no baseline is baked into the image; POST /baseline/load or /baseline/fit at startup, so the
# reference travels with the model artefact rather than with the container tag
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["uvicorn", "driftwatch.service:app", "--host", "0.0.0.0", "--port", "8000"]
