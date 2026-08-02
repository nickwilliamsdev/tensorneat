#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

RUN ldconfig
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Create one stable virtual env in the image and install tooling once.
RUN uv python install 3.12 && \
    uv venv --python 3.12 /opt/venv && \
    uv pip install --python /opt/venv/bin/python \
      "marimo==0.16.5" \
      "jax[cuda13]==0.7.2" \
      "numpy==2.3.3" \
      "plotly==6.3.0" \
      "opencv-python-headless==4.12.0.88" \
      "tqdm==4.67.1"

EXPOSE 8080

# Dev default: open a shell. Start marimo or scripts manually.
CMD ["bash"]