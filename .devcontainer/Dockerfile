# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.177.0/containers/ubuntu/.devcontainer/base.Dockerfile

# [Choice] Ubuntu version: bionic, focal
ARG VARIANT="focal"
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

# Install Python
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y install --no-install-recommends python3-pip && \
    pip3 install numpy

# Install opencv
RUN apt install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx && \
    pip3 install opencv-contrib-python

# Install tesseract
RUN apt install -y tesseract-ocr && \
    pip3 install pytesseract