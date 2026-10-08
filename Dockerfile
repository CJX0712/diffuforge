# DiffuForge — reproducible CPU image.
#   docker build -t diffuforge:0.1.0 .
#   docker run --rm diffuforge:0.1.0            # runs the benchmark CLI
FROM python:3.13-slim

LABEL org.opencontainers.image.title="DiffuForge" \
      org.opencontainers.image.description="NFE-budgeted diffusion sampling with the DiffuFuse sampler" \
      org.opencontainers.image.authors="晨星 (CJX0712)" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /app

COPY requirements.lock.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.lock.txt

COPY pyproject.toml README.md LICENSE ./
COPY diffuforge ./diffuforge
COPY tests ./tests
COPY examples ./examples

RUN python -m pip install --no-cache-dir -e .

# Run the test suite as part of the build so a broken image never ships.
RUN python -m pytest -q

ENTRYPOINT ["python", "-m", "diffuforge.cli"]
CMD ["--out", "/app/benchmark.json"]
