# VIS-MAE
repository for paper "VIS-MAE: An Efficient Self-supervised Learning Approach on Medical Image Segmentation and Classification" accepted in Machine Learning in Medical Imaging workshop of 2024 MICCAI.

# Dataset preparation for pre-training and fine-tuning
For VIS-MAE pretrained model weight training, prepare a directory containing all upstream images.

For the downstream tasks of segmentation or classification, prepare train/validation/test csv files containing two columns ['image','mask'] or ['image','label']. During the finetuning training, send the directory containing the train/validation/test csv files.  
# Generating pretrained MAE-based model weights

use the following command line:

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_pretrain.py --output_dir xxx --log_dir ./output/xxx --data_path xxx --batch_size 640 --epochs 800

# Appyly VIS-MAE pretrained weight for fine-tuning on segmentation or classification tasks
command line for segmentation:
CUDA_VISIBLE_DEVICES=0  python run_finetune_segmentation.py  --output_dir xxx --input_size 320 --fold_num 1  --data_path xxx --batch_size 32 --epochs 150  --task xxx  --transform --optimizer adamw --lr 0.001  --num_classes *number of segmentation labels+1* --warmup_epochs 40   --finetune pretrain-maeall.pth

command line for classification:

# Evaluation of fine-tuned models on test dataset

CUDA_VISIBLE_DEVICES=0  python run_finetune_segmentation.py  --output_dir ./output/ --input_size 320 --fold_num 1  --data_path xxx --batch_size 1 --epochs 150  --task xxx  --eval --optimizer adamw --lr 0.001  --num_classes *number of segmentation labels+1* --warmup_epochs 40   --finetune xxx finetuned model weight xxx
