FROM python:3.12-slim

### SET ENVIRONMENT
USER root
RUN mkdir /workspace
RUN useradd -m -s /bin/bash -u 1000 magus

### INSTALL SUDO AND SET UP PERMISSIONS
RUN apt-get update && apt-get install -y sudo
RUN echo "magus ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

### DEFINE WORK ENVIRONMENT
RUN mkdir -p /workspace/what-lives/
RUN chown -R magus:magus /workspace

### INSTALL DEPENDENCIES
RUN apt-get update && apt-get install -y \
  build-essential \
  libpq-dev \
  wget \
  cowsay \
  python3-pip

### SET ENVIRONMENT VARIABLES
ENV PATH="/usr/games:${PATH}"
ENV PATH="/home/magus/.local/bin:${PATH}"
# ENV PYTHONPATH="/workspace/what-lives:${PYTHONPATH}"
ENV PYTHONPATH="/workspace/what-lives"
WORKDIR /workspace/what-lives

### COPY CONTENTS
COPY . /workspace/what-lives/
RUN chown -R magus:magus /workspace/what-lives/

### INSTALL LIBRARIES
USER magus
RUN pip install --upgrade pip && \
  python3 -m pip install -U -r /workspace/what-lives/requirements.txt && \
  rm /workspace/what-lives/requirements.txt

### SET DEFAULT USER AND WORKING DIRECTORY
USER magus
WORKDIR /workspace/what-lives

ENTRYPOINT ["/bin/bash", "-c", "cowsay namaste && exec /bin/bash"]

########################
