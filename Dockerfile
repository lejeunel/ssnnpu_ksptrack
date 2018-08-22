FROM ubuntu:18.04
MAINTAINER Laurent Lejeune <laurent.lejeune@artorg.unibe.ch>

#Install basic tools
RUN apt-get update && \
    apt-get install -y git cmake vim libboost-all-dev python3-pip &&\
    rm -rf /var/lib/apt/lists/*

