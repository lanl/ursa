FROM ghcr.io/astral-sh/uv:0.9.0-bookworm

# Set working dir
WORKDIR /app

ENV PATH=/root/.local/bin:$PATH
COPY dist/*.whl /wheelhouse/
COPY requirements.txt /app
# RUN uv init -p 3.12
RUN uv pip install --system --break-system-packages -r /app/requirements.txt
RUN uv pip install --system --break-system-packages --no-deps `ls /wheelhouse/*.whl`
RUN ursa --version

# Set default directory to /workspace  
WORKDIR /mnt/workspace
