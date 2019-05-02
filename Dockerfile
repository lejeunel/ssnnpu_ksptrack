FROM nvidia/cuda:9.2-runtime-ubuntu18.04
MAINTAINER Laurent Lejeune <laurent.lejeune@artorg.unibe.ch>

ARG DEBIAN_FRONTEND=noninteractive

#Install basic tools
RUN apt-get update && \
    apt-get install -y build-essential curl tmux git htop cmake vim python3-pip python3-tk libsm6 libxext6 libboost-all-dev psmisc zsh rake libssl-dev zlib1g-dev libbz2-dev libffi-dev libdb-dev libexpat-dev libreadline-dev libsqlite3-dev wget libncurses5-dev libncursesw5-dev xz-utils&&\
    rm -rf /var/lib/apt/lists/*

#Install boost_ksp
RUN cd /home && git clone https://github.com/lejeunel/boost_ksp.git \
  && cd boost_ksp \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j 4 \
  && python3 src/setup.py install \
  && python3 ../demo/demo.py \ 
  && rm -rf /home/boost_ksp

#Install SLICsupervoxels
RUN cd /home && git clone https://github.com/lejeunel/SLICsupervoxels.git \
  && cd SLICsupervoxels \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j 4 \
  && python3 src/setup.py install \
  && rm -rf /home/SLICsupervoxels

#Install pyflow
RUN cd /home && git clone https://github.com/Illumina/pyflow.git \
  && cd pyflow/pyflow \
  && python3 setup.py build install 

#Install KSPTrack
RUN cd /home && git clone https://github.com/lejeunel/ksptrack.git \
  && cd ksptrack \
  && pip3 install -r requirements.txt \
  && pip3 install -e .
