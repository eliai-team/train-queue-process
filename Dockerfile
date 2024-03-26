# Base image
FROM python:3.10.9-slim
# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ADD . .
# Set the working directory
WORKDIR /

# COPY builder/setup.sh /setup.sh
# RUN bash /setup.sh

RUN apt-get update && \
    apt install -y \
    fonts-dejavu-core rsync git jq moreutils aria2 wget libgoogle-perftools-dev procps libgl1 libglib2.0-0 && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    LD_PRELOAD=libtcmalloc.so \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/cache --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# Install Python dependencies (Worker Template)
# COPY README-ja.md README.md
COPY requirements.txt ./requirements.txt
COPY basemodel.safetensors ./basemodel.safetensors
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade -r ./requirements.txt --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install -U xformers --index-url https://download.pytorch.org/whl/cu118 && \
    pip install prodigyopt && \
    pip install -U dadaptation


RUN wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;
# Add src files (Worker Template)
# ADD src /sd-scripts

# WORKDIR /sd-scripts
COPY run.sh ./run.sh
RUN chmod +x ./run.sh


CMD ./run.sh