FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

ENV CONDA_ENV_NAME=video_benchmark
ENV PYTHON_VERSION=3.8
ENV HOME=/home

RUN apt update && apt install -y vim \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   wget \
                   && rm -rf /var/lib/apt/lists


# Create a Python 3.8 environment
# Install Miniconda and create main env
ADD https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh  miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /conda \
    && echo export PATH=/conda/bin:$PATH >> .bashrc \
    && rm miniconda3.sh
ENV PATH="/conda/bin:${PATH}"

# Switch to bash shell
SHELL [ "/bin/bash", "-c" ]

COPY environment.yaml ./
RUN conda env create --name ${CONDA_ENV_NAME} --file environment.yaml && rm environment.yaml

ENV PATH="/conda/envs/${CONDA_ENV_NAME}/bin:$PATH"

RUN mkdir -p /data
RUN mkdir /workspace
# Set working directory
WORKDIR /workspace/

RUN chmod -R 777 $HOME/
# Set ${CONDA_ENV_NAME} to default virutal environment
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

