FROM nvidia/cuda:11.3.1-base-ubuntu20.04
MAINTAINER Laurent Lejeune <me@lejeunel.org>

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.6
ARG OPENCV_VERSION=4.1.1

#Install basic tools
RUN apt-get update && \
    apt-get install -y build-essential curl git cmake python3-pip python3-tk graphviz sudo &&\
    rm -rf /var/lib/apt/lists/*

#Install pyksp
RUN cd /home && git clone --recurse-submodules https://github.com/lejeunel/pyksp.git \
  && cd pyksp \
  && sudo pip install .
