#!/bin/sh


eval "$(pyenv init -)"
pyenv activate my-3.5


sed -i "s/learn_dir_pre = '.*'/learn_dir_pre = 'slitlamp'/g" learning_exp.py
python learning_exp.py

sed -i "s/learn_dir_pre = '.*'/learn_dir_pre = 'brain'/g" learning_exp.py
python learning_exp.py

#sed "s/learn_dir_pre = '.*'/learn_dir_pre = 'tweezer'/g" learning_exp.py | less
#python learning_exp.py

