FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn pydantic openenv-core --no-deps
RUN pip install --no-cache-dir requests pyyaml websockets httpx

COPY . .

ENV PYTHONPATH=/app:/app/server

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
