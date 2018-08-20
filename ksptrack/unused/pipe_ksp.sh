#!/bin/sh

eval "$(pyenv init -)"
pyenv activate my-3.5

cd ~/Documents/labeling
pip install -r requirements.txt
python pipe_ksp.py
