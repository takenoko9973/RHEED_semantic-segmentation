# https://github.com/anibali/docker-pytorch/tree/master

FROM ubuntu:22.04 AS installer

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

RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
        git \
        wget \
        p7zip-full \
        fontconfig \
        ttf-mscorefonts-installer && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG FONT_FOLDER="/usr/share/fonts"
WORKDIR ${FONT_FOLDER}
RUN wget -q https://github.com/trueroad/HaranoAjiFonts/raw/refs/heads/master/HaranoAjiGothic-Bold.otf; \
    wget -q https://github.com/trueroad/HaranoAjiFonts/raw/refs/heads/master/HaranoAjiMincho-Bold.otf; \
    wget -q https://github.com/trueroad/HaranoAjiFonts/raw/refs/heads/master/HaranoAjiGothic-Regular.otf; \
    wget -q https://github.com/trueroad/HaranoAjiFonts/raw/refs/heads/master/HaranoAjiMincho-Regular.otf; \
    wget -q https://github.com/trueroad/HaranoAjiFonts/raw/refs/heads/master/HaranoAjiGothic-Light.otf; \
    wget -q https://github.com/trueroad/HaranoAjiFonts/raw/refs/heads/master/HaranoAjiMincho-Light.otf; \
    fc-cache -fv;

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
