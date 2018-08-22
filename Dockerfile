FROM tensorflow/tensorflow:latest-gpu-py3
MAINTAINER Laurent Lejeune <laurent.lejeune@artorg.unibe.ch>
#Install basic tools
RUN apt-get update && \
    apt-get install -y git libbz2-dev cmake vim python3-tk wget &&\
    rm -rf /var/lib/apt/lists/*


# Download boost, untar, setup install with bootstrap 
# and then install
RUN cd /home && wget http://downloads.sourceforge.net/project/boost/boost/1.66.0/boost_1_66_0.tar.gz \
  && tar xfz boost_1_66_0.tar.gz \
  && rm boost_1_66_0.tar.gz \
  && cd boost_1_66_0 \
  && sh bootstrap.sh --with-libraries=python,log --with-python=/usr/local/bin/python3 --with-python-version=3.5 --with-python-root=/usr/local/lib/python3.5 \
  && ./b2 -j 4 install \
  && cd /home \
&& rm -rf boost_1_66_0

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
  && pip3 install -e .
