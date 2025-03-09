# https://github.com/anibali/docker-pytorch/tree/master

FROM ubuntu:22.04 as installer

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities.
RUN apt-get update && apt-get install -y \
        curl \
        ca-certificates \
        bzip2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install Micromamba.
RUN curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local bin/micromamba

# ==================================================

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN rm -f /etc/apt/sources.list.d/*.list

RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
        git \
        p7zip-full && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a working directory.
RUN mkdir /app
WORKDIR /app

COPY --from=installer /usr/local/bin/micromamba /usr/local/bin/micromamba
ENV MAMBA_EXE=/usr/local/bin/micromamba \
    MAMBA_ROOT_PREFIX=/home/user/micromamba \
    CONDA_PREFIX=/home/user/micromamba \
    PATH=/home/user/micromamba/bin:$PATH

# Set up the base Conda environment by installing PyTorch and friends.
COPY ./environment.yml /app/
RUN micromamba create -qy -n base -f /app/environment.yml \
    && rm -f /app/environment.yml \
    && micromamba shell init --shell=bash \
    && micromamba clean -qya

# # Fix for https://github.com/pytorch/pytorch/issues/97041
RUN ln -s "$CONDA_PREFIX/lib/libnvrtc.so.*.*.*" "$CONDA_PREFIX/lib/libnvrtc.so"

CMD ["bash"]
