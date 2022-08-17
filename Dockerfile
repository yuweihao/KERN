FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /kern


RUN apt update -y && \
    apt install -y -qq wget graphviz 

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda init

COPY environment.yml .

RUN conda env create -f ./environment.yml
