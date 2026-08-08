FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser dashboard ./dashboard
COPY --chown=appuser:appuser data/sample_payloads ./data/sample_payloads
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser models/payguard_calibrated_policy.joblib ./models/payguard_calibrated_policy.joblib

RUN python -c "from api.model_loader import load_policy; load_policy()"

USER appuser

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
