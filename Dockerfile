FROM ghcr.io/astral-sh/uv:0.11-python3.12-trixie
# FROM ghcr.io/astral-sh/uv:0.11.7-python3.12-bookworm
# FROM astral/uv:0.11.7-python3.12-bookworm

# Set working dir
WORKDIR /app

ENV PATH=/root/.local/bin:$PATH
COPY dist/*.whl /wheelhouse/
COPY requirements.txt /app
RUN uv pip install --system --break-system-packages -r /app/requirements.txt
RUN uv pip install --system --break-system-packages "$(ls /wheelhouse/*.whl)"
# RUN uv pip install --system --break-system-packages --no-deps `ls /wheelhouse/*.whl`
RUN ursa --version

# Set default directory to /workspace  
WORKDIR /mnt/workspace
