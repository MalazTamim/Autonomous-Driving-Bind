FROM ubuntu:20.04

# Build arguements cann be passed to docker build via --build-arg USER=<username> --build-arg UID=<user-id>
ARG USER=user
ARG UID=1234

# Needed to disable install dialogs during build
ENV DEBIAN_FRONTEND=noninteractive

# Update apt package cache
RUN apt update

# Install dependencies
RUN apt install -y python3 python3-pip git ffmpeg libsm6 libxext6

# Update pip
RUN python3 -m pip install --upgrade pip

# Install pip requirements sequentially
COPY ./requirements.txt /tmp/requirements.txt
# RUN cat /tmp/requirements.txt | xargs -n 1 -L 1 pip3 install
RUN python3 -m pip install -r /tmp/requirements.txt

# Add user
RUN adduser --disabled-password --gecos '' --uid ${UID} ${USER}
RUN apt install -y sudo
RUN adduser ${USER} sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
