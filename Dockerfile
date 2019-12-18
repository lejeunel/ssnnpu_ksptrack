FROM nvidia/cuda:9.2-runtime-ubuntu18.04
MAINTAINER Laurent Lejeune <laurent.lejeune@artorg.unibe.ch>

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.6
ARG OPENCV_VERSION=4.1.1

#Install basic tools
RUN apt-get update && \
    apt-get install -y build-essential curl tmux git htop cmake vim python3-pip python3-tk libsm6 libxext6 libboost-all-dev psmisc zsh rake libssl-dev zlib1g-dev libbz2-dev libffi-dev libdb-dev libexpat-dev libreadline-dev libsqlite3-dev wget libncurses5-dev libncursesw5-dev xz-utils ranger slurm-client&&\
    rm -rf /var/lib/apt/lists/*

# zsh stuff
RUN cd /usr/local/share/zsh && \
chmod -R 755 ./site-functions&&\
chown -R root:root ./site-functions

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


# Install all dependencies for OpenCV
RUN    apt-get -y update --fix-missing && \
    apt-get -y install --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        $( [ ${PYTHON_VERSION%%.*} -ge 3 ] && echo "python${PYTHON_VERSION%%.*}-distutils" ) \
        wget \
        unzip \
        cmake \
        libtbb2 \
        gfortran \
        apt-utils \
        pkg-config \
        checkinstall \
        qt5-default \
        build-essential \
        libatlas-base-dev \
        libgtk2.0-dev \
        libavcodec57 \
        libavcodec-dev \
        libavformat57 \
        libavformat-dev \
        libavutil-dev \
        libswscale4 \
        libswscale-dev \
        libjpeg8-dev \
        libpng-dev \
        libtiff5-dev \
        libdc1394-22 \
        libdc1394-22-dev \
        libxine2-dev \
        libv4l-dev \
        libgstreamer1.0 \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-0 \
        libgstreamer-plugins-base1.0-dev \
        libglew-dev \
        libpostproc-dev \
        libeigen3-dev \
        libtbb-dev \
        zlib1g-dev \
        libsm6 \
        libxext6 \
        libxrender1 \
    && \

# install python dependencies
    sysctl -w net.ipv4.ip_forward=1 && \
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \
    pip${PYTHON_VERSION} install numpy && \

# Install OpenCV
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip -O opencv.zip --progress=bar:force:noscroll && \
    unzip -q opencv.zip && \
    mv /opencv-$OPENCV_VERSION /opencv && \
    rm opencv.zip && \
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip -O opencv_contrib.zip --progress=bar:force:noscroll && \
    unzip -q opencv_contrib.zip && \
    mv /opencv_contrib-$OPENCV_VERSION /opencv_contrib && \
    rm opencv_contrib.zip && \

# Prepare build
    mkdir /opencv/build && \
    cd /opencv/build && \
    cmake \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D BUILD_PYTHON_SUPPORT=ON \
      -D BUILD_DOCS=ON \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_TESTS=OFF \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
      -D BUILD_opencv_python3=$( [ ${PYTHON_VERSION%%.*} -ge 3 ] && echo "ON" || echo "OFF" ) \
      -D BUILD_opencv_python2=$( [ ${PYTHON_VERSION%%.*} -lt 3 ] && echo "ON" || echo "OFF" ) \
      -D PYTHON${PYTHON_VERSION%%.*}_EXECUTABLE=$(which python${PYTHON_VERSION}) \
      -D PYTHON_DEFAULT_EXECUTABLE=$(which python${PYTHON_VERSION}) \
      -D BUILD_EXAMPLES=OFF \
      -D WITH_IPP=OFF \
      -D WITH_FFMPEG=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_V4L=ON \
      -D WITH_LIBV4L=ON \
      -D WITH_TBB=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
      -D WITH_CUBLAS=ON \
      -D WITH_NVCUVID=ON \
      -D ENABLE_FAST_MATH=1 \
      -D ENABLE_PRECOMPILED_HEADERS=OFF \
      .. \
    && \

# Build, Test and Install
    cd /opencv/build && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \

# cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /opencv /opencv_contrib /var/lib/apt/lists/* && \

# Set the default python and install PIP packages
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \

# Call default command.
    python --version && \
    python -c "import cv2 ; print(cv2.__version__)"


#Install requirements for gpb
RUN apt-get update && \
    apt-get install -y libarpack2-dev libparpack2-dev libarpack++2-dev&&\
    rm -rf /var/lib/apt/lists/*

#Install gpb
RUN cd /home && git clone --recurse-submodules https://github.com/lejeunel/gPb-GSoC.git \
  && cd gPb-GSoC \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j 4 \
  && python3 src/setup.py install \
  && rm -rf /home/gPb-GSoC

# Install all dependencies for OpenCV
RUN    apt-get -y update --fix-missing && \
    apt-get -y install --no-install-recommends \
        g++ \
        libpng-dev \
        git \
        libtiff-dev \
    && \

# Install OpenCV
        git clone https://github.com/InsightSoftwareConsortium/ITK.git && \
        cd ITK && \

# Prepare build
        mkdir build && \
        cd build && \
        cmake .. &&\

# Build, Test and Install
    make -j$(nproc) && \
    make install && \
    ldconfig && \

# cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /ITK /var/lib/apt/lists/*

# Install Shogun
RUN    cd / && \
    apt-get -y update --fix-missing && \
    apt-get -y install --no-install-recommends \
        cmake \
        make \
        git \
    && \

    git clone https://github.com/shogun-toolbox/shogun.git && \
    cd shogun && \
    git submodule update --init && \

# Prepare build
    mkdir build && \
    cd build && \
    cmake -DBUILD_EXAMPLES=OFF -DBUILD_META_EXAMPLES=OFF .. &&\

# Build, Test and Install
    make -j$(nproc) && \
    make install && \
    make clean && \

# cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /shogun /var/lib/apt/lists/*

