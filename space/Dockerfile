FROM ubuntu:24.04
ARG LLAMA_TAG=b12110
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ca-certificates curl libgomp1 libssl3 zlib1g libbrotli1 libzstd1 \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/llama && curl -fsSL \
      https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz \
    | tar -xz -C /opt/llama --strip-components=1 \
    && /bin/sh -c 'LD_LIBRARY_PATH=/opt/llama /opt/llama/llama-server --version'
RUN pip3 install --no-cache-dir --break-system-packages gradio==6.32.0 huggingface_hub
RUN useradd -m -u 1000 user || true
USER 1000
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface LLAMA_DIR=/opt/llama PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=1000 . .
EXPOSE 7860
CMD ["python3", "app.py"]
