import os

os.system('python single_ksp_gpu.py --in-path /home/laurent.lejeune/medical-labeling/Dataset00 --out-path /home/laurent.lejeune/medical-labeling/Dataset00 --entrance-masks-path /home/laurent.lejeune/medical-labeling/unet_region/runs/Dataset00_2019-04-10_11-38-11/Dataset00/entrance_masks --bag-t 500 --bag-jobs 4')
