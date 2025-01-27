ARG BASE=nvidia/cuda:11.8.0-base-ubuntu22.04
FROM ${BASE}

# Install OS dependencies:
RUN apt-get update && apt-get upgrade -y --fix-missing
RUN apt-get install -y --no-install-recommends \
    gcc g++ \
    make \
    python3 python3-dev python3-pip python3-venv python3-wheel \
    espeak-ng libsndfile1-dev \
    wget unzip bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Major Python Dependencies:
RUN pip3 install llvmlite --ignore-installed
RUN pip3 install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN rm -rf /root/.cache/pip

WORKDIR /root

# Copy Dependency Lock Files:
COPY \
    Makefile \
    pyproject.toml \
    setup.py \
    requirements.dev.txt \
    requirements.ja.txt \
    requirements.notebooks.txt \
    requirements.txt \
    /root/

# Install Project Dependencies
# Separate stage to limit re-downloading:
RUN pip install \
    -r requirements.txt \
    -r requirements.dev.txt \
    -r requirements.ja.txt \
    -r requirements.notebooks.txt

RUN pip3 install gdown

# Copy TTS repository contents:
COPY . /root

ENV NUMBA_CACHE_DIR=/tmp

# # Set environment variable for fsspec cache directory
# ENV FSSPEC_CACHEDIR=/tmp/fsspec_cache

# # Create the cache directory
# RUN mkdir -p /tmp/fsspec_cache

# Installing the TTS package itself:
RUN make install

RUN chmod +777 -R /