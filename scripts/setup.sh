#!/usr/bin/env bash
echo "Installing resources for AMS"
source scripts/folder_setup.sh


if ! command -v conda > /dev/null
  then echo "Missing conda, please install"
  exit 1
fi

if [[ -z ${CONDA_EXE} ]]
then
    echo "Need to set CONDA_EXE environment variable"
    exit 1
fi
# install everything into appropriate conda environment
conda_folder=$(realpath "$(dirname $CONDA_EXE)/..")
source ${conda_folder}/etc/profile.d/conda.sh || { echo "Missing conda.sh"; exit 1; }

conda activate ams-env
if [[ $? -ne 0 ]]
then
    echo "Need to build conda environment ams-env"
    conda env create -f environment.yml
fi

# Create base environment
conda activate ams-env

# SciSpacy model
mkdir -p ${RESOURCES}
pushd ${RESOURCES}
# use wget to download, pip seems to mess it up somehow
wget https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
pip install en_core_sci_lg-0.2.4.tar.gz

# SciBert model
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar
tar xvf scibert_scivocab_uncased.tar
pushd scibert_scivocab_uncased
tar xvf weights.tar.gz
# required to properly load
cp bert_config.json config.json
popd
popd

# Download Kaggle dataset
mkdir -p ${DATA}
pushd ${DATA}
wget https://archive.org/download/meta-kaggle/meta-kaggle.zip
unzip meta-kaggle.zip -d meta-kaggle/
popd


# install task-spooler
# https://vicerveza.homeunix.net/~viric/soft/ts/
if [[ $(uname) == "Darwin" ]]
then
    brew install task-spooler
else
    sudo apt-get install -y task-spooler
fi

# install custom patched TPOT version
# https://github.com/EpistasisLab/tpot/pull/1024
pushd tpot
pip install -e .
popd
