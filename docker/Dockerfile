# Base image
FROM nvidia/cudagl:11.1-devel-ubuntu20.04

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install conda
WORKDIR /tmp
RUN wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -bfp /opt/conda
RUN rm /tmp/Miniconda3-latest-Linux-x86_64.sh
RUN /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include
RUN /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN bash cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Conda environment
RUN conda create -n habitat python=3.8 cmake=3.14.0

# Activate the habitat conda environment
RUN echo "source activate habitat" >> /root/.bashrc
RUN ["/bin/bash", "-c", "source activate habitat"]

# Install tzdata
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install pytorch
RUN conda install pytorch==1.6.0 torchvision=0.7.0 cudatoolkit=10.2.89 -c pytorch

WORKDIR /root/home/ppo_for_beginners
