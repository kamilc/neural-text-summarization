#!/bin/bash

apt-get upate
apt-get install unzip

conda install -y jupyterlab hypothesis pandas transformers conda-forge spacy tqdm ipywidgets fastparquet
python -m spacy download en_core_web_lg scikit-learn
pip install plotnine hickle
