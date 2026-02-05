FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r relay && useradd -r -g relay -m relay

WORKDIR /app

COPY relay/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=relay:relay relay/ ./relay/

USER relay

EXPOSE 8002

CMD ["uvicorn", "relay.main:app", "--host", "0.0.0.0", "--port", "8002"]


