FROM python:3.11
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install supercronic
RUN curl -fsSL -o /usr/local/bin/supercronic \
    https://github.com/aptible/supercronic/releases/download/v0.2.29/supercronic-linux-amd64 \
 && chmod +x /usr/local/bin/supercronic

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

RUN uv pip install --system -e ".[dev]"

COPY docker/entrypoint.sh /app/docker/entrypoint.sh
COPY docker/crontab /app/docker/crontab
RUN sed -i 's/\r$//' /app/docker/entrypoint.sh && chmod +x /app/docker/entrypoint.sh

ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]