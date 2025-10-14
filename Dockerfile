FROM ghcr.io/astral-sh/uv:0.9.0-bookworm

# Set working dir
WORKDIR /app

ENV PATH=/root/.local/bin:$PATH
COPY dist/*.whl /wheelhouse/
RUN uv init -p 3.12
RUN uv add `ls /wheelhouse/*.whl`
ENV PATH=/app/.venv/bin:$PATH
RUN ursa version

# Set environment in /app as default uv environment
ENV UV_PROJECT=/app

# Set default directory to /workspace  
WORKDIR /mnt/workspace
