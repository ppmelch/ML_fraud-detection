# Backend API image for the transaction anomaly-detection dashboard.
#
# Build context is the REPO ROOT (see docker-compose.yml), not backend/,
# because backend/src/modeling/config.py resolves two runtime paths outside
# backend/ by walking up from its own file location:
#   - GEOJSON_PATH  -> <repo root>/frontend/data/us-states.json
#   - ARTIFACTS_DIR -> <repo root>/backend/artifacts
# Preserving the same backend/ + frontend/ sibling layout inside the image
# keeps that path arithmetic correct without touching application code.
#
# Build:
#   docker build -f backend/Dockerfile -t fraud-backend .
# Run:
#   docker run -p 8000:8000 fraud-backend

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dependency layer cached on requirements.txt alone, so editing application
# code never forces a reinstall of pandas/scikit-learn/shap.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Application code, precomputed JSON artifacts and the trained model bundle.
COPY backend/ backend/

# Only the one frontend file the backend itself reads (the US states
# GeoJSON, served back out through /api/map/geojson). The rest of frontend/
# is served by the frontend container, not this one.
COPY frontend/data/us-states.json frontend/data/us-states.json

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "backend.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
