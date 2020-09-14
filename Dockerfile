FROM ubuntu:18.04

# Basic prereqs
RUN apt update
RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y zip
RUN apt-get install -y vim
RUN apt-get update
RUN apt-get install -y gcc


# install miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/root/miniconda3/bin/:${PATH}

ADD . ams/
WORKDIR ams/

# remove torch (errors out)
# https://discuss.pytorch.org/t/memory-error-when-installing-pytorch/8027/6
# instead install afterwards with pip so we can use the --no-cache-dir
RUN egrep -v "torch|spacy-transformers" environment.yml > environment_patched.yml
RUN conda env create -f environment_patched.yml
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
  conda activate ams-env && \
  pip install --no-cache-dir torch==1.3.1 && \
  pip install --no-cache-dir spacy-transformers==0.5.1 && \
  pip install --no-cache-dir torchcontrib==0.0.2

RUN cat /root/miniconda3/etc/profile.d/conda.sh >> ~/.bashrc

ENTRYPOINT bash
