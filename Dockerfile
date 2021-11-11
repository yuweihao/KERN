FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONPATH=/kern

WORKDIR /kern

COPY environment.yml .

RUN apt update -y && \
    apt upgrade -y && \
    apt install -y wget && \
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda init

RUN conda env create -f ./environment.yml
