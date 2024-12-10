# VIS-MAE
repository for paper "VIS-MAE: An Efficient Self-supervised Learning Approach on Medical Image Segmentation and Classification" accepted in Machine Learning in Medical Imaging workshop of 2024 MICCAI.

# Dataset preparation for pre-training and fine-tuning
For VIS-MAE pretrained model weight training, prepare a directory containing all upstream images.

For the downstream tasks of segmentation or classification, prepare a csv file containing two columns ['image','mask'] or ['image','label'].
# Generating pretrained MAE-based model weights

use the following command line:

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_pretrain.py --output_dir xxx --log_dir ./output/xxx --data_path xxx --batch_size 640 --epochs 800
