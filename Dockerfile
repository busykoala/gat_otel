FROM python:3.11-slim

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="${PATH}:/root/.local/bin"

RUN poetry config virtualenvs.create false

COPY . .

RUN poetry install --no-interaction --no-ansi --no-dev

RUN useradd -m appuser
RUN chown -R appuser:appuser /usr/src/app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "otel_backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
