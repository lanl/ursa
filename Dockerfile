# https://docs.astral.sh/uv/guides/integration/docker/
# NOTE: This is compatible with docker and charliecloud
FROM ghcr.io/astral-sh/uv:0.9.0-bookworm

# Get current git tag
ARG GIT_TAG

# Set working dir
WORKDIR /app

# Get essentials
RUN apt update && apt install -y build-essential curl ca-certificates git

RUN git config --global init.defaultBranch main
RUN git config --global user.email "ursa@fake-domain.com"
RUN git config --global user.name "ursa-bot"

# ursa directories
COPY .gitignore /app/.gitignore
COPY examples /app/examples
COPY docs /app/docs
COPY src /app/src

# ursa files
COPY LICENSE /app
COPY README.md /app
COPY pyproject.toml /app
COPY uv.lock /app

# Sync ursa environment. Use git to inform version.
RUN uv python pin 3.12
RUN git init 
RUN git add -A
RUN git commit -m 'init'
RUN git tag ${GIT_TAG}
RUN uv sync --no-cache --all-groups --no-dev --locked
RUN cp /app/.venv/bin/ursa /bin
RUN ursa version

# Set environment in /app as default uv environment
ENV UV_PROJECT=/app

# Set default directory to /workspace  
WORKDIR /mnt/workspace
