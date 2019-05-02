Bootstrap: docker
From: nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04

%environment

# Update list of packages then upgrade them
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y upgrade
    
# Install dependencies
apt-get install -y htop libbz2-dev vim python3-pip\
        python3-tk wget libsm6 libboost-all-dev psmisc zsh rake\
        apt-get install -y libssl-dev zlib1g-dev libffi-dev\
        libssl-dev libdb-dev libexpat-dev libreadline-dev libsqlite3-dev\
        wget curl libncurses5-dev libncursesw5-dev xz-utils
apt-get install -y --no-install-recommends build-essential cmake git curl vim \
        ca-certificates libjpeg-dev libpng-dev
apt-get install -y tmux vim wget 

# More dependencies/useful software from system package manager
apt-get install -v libopenblas-dev libfreetype7-dev libpng12-dev \
    libglib2.0-0 libxext6 libxrender1
    
# Dense Flow dependencies
apt-get install -y libzip-dev

# OpenCV build dependencies not already installed  
apt-get install -y checkinstall yasm libjpeg-dev libjpeg8-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libdc1394-22-dev \
    libv4l-dev python-dev python-numpy \
    libtbb-dev libqt4-dev libgtk2.0-dev libfaac-dev libmp3lame-dev \
    libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev \
    libxvidcore-dev x264 v4l-utils libgtk-3-dev
    
# Add PPA for ffmpeg
add-apt-repository -y ppa:jonathonf/ffmpeg-4
apt-get update
apt-get install -y ffmpeg

# More utilities
apt-get install -y graphviz libfreetype6 libfreetype6-dev \
    libgraphviz-dev liblapack-dev swig libxft-dev libxml2-dev \
    libxslt-dev zlib1g-dev
 
# Gnuplot
apt-get install -y gnuplot-x11
    
# Clean up
apt-get -y autoremove
rm -rvf /var/lib/apt/lists/*
