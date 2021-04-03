# Base image
FROM nvidia/cudagl:11.1-devel-ubuntu20.04

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install python3.8 and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip3 install \
    absl-py==0.10.0 \
    astor==0.8.1\
    atari-py==0.2.6\
    Box2D==2.3.10\
    certifi==2020.6.20 \
    cloudpickle==1.3.0 \
    cycler==0.10.0 \
    future==0.18.2 \
    gast==0.4.0 \
    google-pasta==0.2.0 \
    grpcio==1.31.0 \
    gym==0.17.2 \
    h5py==2.10.0 \
    importlib-metadata==1.7.0 \
    joblib==0.16.0 \
    Keras-Applications==1.0.8 \
    Keras-Preprocessing==1.1.2 \
    kiwisolver==1.2.0 \
    Markdown==3.2.2 \
    matplotlib==3.3.1 \
    numpy==1.19.1 \
    opencv-python==4.4.0.42 \
    pandas==1.1.1 \
    Pillow==7.2.0 \
    protobuf==3.13.0 \
    pyglet==1.5.0 \
    pyparsing==2.4.7 \
    python-dateutil==2.8.1 \
    pytz==2020.1 \
    scipy==1.5.2 \
    six==1.15.0 \
    stable-baselines==2.10.1 \
    tensorboard==1.5.0 \
    tensorflow==1.5.0 \
    tensorflow-estimator==1.5.0 \
    termcolor==1.1.0 \
    torch==1.6.0 \
    Werkzeug==1.0.1 \
    wrapt==1.12.1 \
    zipp==3.1.0


WORKDIR /root/home/src
