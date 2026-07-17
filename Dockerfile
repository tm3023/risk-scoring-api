FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models_store/ ./models_store/

EXPOSE 8000

# --host 0.0.0.0 is required for the container's network to be reachable
# from outside; most PaaS platforms also expect the app to read the PORT
# env var, so we fall back to 8000 for local/docker-run use.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
