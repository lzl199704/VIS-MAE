# VIS-MAE
repository for paper "VIS-MAE: An Efficient Self-supervised Learning Approach on Medical Image Segmentation and Classification" accepted in Machine Learning in Medical Imaging workshop of 2024 MICCAI.

# Generating pretrained MAE-based model weights

use the following command line:

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_pretrain.py --output_dir xxx --log_dir ./output/xxx --data_path xxx --batch_size 640 --epochs 800
