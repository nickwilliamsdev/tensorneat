# (Keep the comments and FROM at the top)
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

RUN ldconfig
COPY --from=ghcr.io/astral-sh/uv:latest	/uv /uvx /bin/

RUN mkdir /app
WORKDIR /app

# Point PATH to an environment OUTSIDE of /app
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Copy pyproject.toml FIRST so we can install dependencies from it
COPY pyproject.toml .

# Create the venv and install everything listed in pyproject.toml, plus test tools
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv/bin/python -e . \
    && uv pip install --python /opt/venv/bin/python "jax[cuda13]==0.7.2" \
    && uv pip install --python /opt/venv/bin/python "numpy==2.3.3" \
    && uv pip install --python /opt/venv/bin/python "plotly==6.3.0" \
    && uv pip install --python /opt/venv/bin/python "opencv-python-headless==4.12.0.88" \
    && uv pip install --python /opt/venv/bin/python "tqdm==4.67.1" \
    && uv pip install --python /opt/venv/bin/python "pytest"

CMD ["bash"]