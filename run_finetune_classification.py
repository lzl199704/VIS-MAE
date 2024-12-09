import argparse
import json
import numpy as np
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import swin_class
from torchvision.transforms import functional as TF
import utils_f
from utils.engine_finetune_classification import train_one_epoch, evaluate, DiceLoss
from datasets import build_pretraining_dataset
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import random
import yacs
import pandas as pd

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label
    
    
class JointTransform:
    def __init__(self, size,apply_transform):
        self.size = size  # Size can be a tuple (width, height) or an integer
        self.do_transform = apply_transform
    def __call__(self, image, mask):
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        image = np.asarray(image)
        mask = np.asarray(mask)
        #x,y = image.shape
        #image = zoom(image, (self.size[0] / x, self.size[1] / y,1), order=3)  # why not 3?
        #mask = zoom(mask, (self.size[0] / x, self.size[1] / y), order=0)
        ###add some other transform methods
        
        if self.do_transform == True:
            if random.random() > 0.5:
                image, mask = random_rot_flip(image, mask)
            elif random.random() > 0.5:
                image, mask = random_rotate(image, mask)
        # Convert the thresholded mask back to a tensor
        image= Image.fromarray(image)
        mask = Image.fromarray(mask)
        return image, mask  
        
def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # common parameters
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--transform', action='store_true')
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--checkpoint_encoder', default='', type=str)
    parser.add_argument('--checkpoint_decoder', default='', type=str)
    parser.add_argument('--data_path', default=r'/data/path', type=str)  # fill in the dataset path here
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # model parameters
    parser.add_argument('--model', default='swin_mae', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--resume', type=str,default='',
                        help='resume path')
    parser.add_argument('--finetune', type=str,default='',
                        help='finetune path')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # optimizer parameters
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # other parameters
    parser.add_argument('--task', default='amos', type=str)
    parser.add_argument('--fold_num', default='0', type=str)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_img', default=False,
                        help='use for test evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def convert_label(input_str):
    if input_str == 'positive' or input_str == 'yes':
        return 1
    else:
        return 0

        
class ClassificationCSVDataset(Dataset):
    def __init__(self,task, root_dir, image_size,fold_num, split="train",add_transform=True):
        self.root_dir = root_dir   ### the csv path containing image list and mask list
        self.dataframe = pd.read_csv(os.path.join(root_dir,split+'_fold'+fold_num+'.csv'))
        self.split = split
        self.task = task
        #self.joint_transform = JointTransform(size=(image_size, image_size),apply_transform=add_transform)  # Example size
        self.do_transform = add_transform
        #no transform for segmentation task. This could be any additional image-specific transform
        
        self.image_size = image_size
        self.images = dataframe['filename'].tolist()
        self.labels = dataframe['label'].tolist()
        #self.images_dir = os.path.join(root_dir, split, "img")
        #self.masks_dir = os.path.join(root_dir, split, "msk")
        self.to_tensor = transforms.ToTensor() 
        #self.images = sorted(os.listdir(self.images_dir))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize to a fixed size
            transforms.ToTensor(),
            # Include normalization if required
        ])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx] #]os.path.join(self.images_dir, self.images[idx])
        label = convert_label(self.labels[idx]) 

        image = Image.open(img_name).convert("RGB")   
        image = TF.resize(image, self.image_size)
        image = np.asarray(image)
        
        if self.do_transform == True:
            if random.random() > 0.5:
                k = np.random.randint(0, 4)
                image = np.rot90(image, k)
                axis = np.random.randint(0, 2)
                image = np.flip(image, axis=axis).copy()
            elif random.random() > 0.5:
                angle = np.random.randint(-20, 20)
                image = ndimage.rotate(image, angle, order=0, reshape=False)
        # Convert the thresholded mask back to a tensor
        image = (image - image.min())/(image.max()-image.min()+1e-6)
        image = (image*255).astype(np.uint8)
        image= Image.fromarray(image)
        
        image = self.transform(image)
        
        #image = torch.tensor(np.asarray(image), dtype=torch.float32).permute(2, 0, 1)
        
        # Convert label to a tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label, img_name
        
        
def main(args):
    utils_f.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils_f.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
        
    # Set model
    model = swin_class.__dict__['swin_class'](num_classes=args.num_classes)
    
    # get dataset
    transform_train = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Set dataset
    #dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    dataset_train = ClassificationCSVDataset(root_dir=args.data_path,image_size = args.input_size, mode="train" ,add_transform=args.transform)
    if args.eval == True:
        dataset_val = ClassificationCSVDataset(root_dir=args.data_path,image_size = args.input_size, mode="test" ,add_transform=False)
    else:
        dataset_val = ClassificationCSVDataset(root_dir=args.data_path,image_size = args.input_size, mode="val" ,add_transform=False)
    if True:  # args.distributed:
        num_tasks = utils_f.get_world_size()
        global_rank = utils_f.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils_f.seed_worker
    )
    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    
    # Log output
    
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(logdir=args.log_dir)
        #log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None
    
    model.to(device)
    model_without_ddp = model

    # Set optimizer
    param_groups = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(param_groups, lr=args.lr)#, weight_decay=5e-2, betas=(0.9, 0.95))  # ???5E-2
    #optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=0.0001)
    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()
    # Create model
    misc.load_model(args=args, model_without_ddp=model_without_ddp)
    
    min_loss = 100.0
    
    
    ### test evaluation
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args.num_classes, args.output_dir)
        #print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['dice_score']:.3f}")
        exit(0)
    
    # Start the training process
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch + 1)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }
        
        
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device,args.num_classes,'')
            #print(f"Dice score of the network on the {len(dataset_val)} val images: {test_stats['loss']:.3f}")
            if min_loss > test_stats["loss"]:
                min_loss = test_stats["loss"]
                
                misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best")

            print(f'Min val loss: {min_loss:.3f}')


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch}
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    arg = get_args_parser()
    arg = arg.parse_args()
    if arg.output_dir:
        Path(arg.output_dir).mkdir(parents=True, exist_ok=True)
    main(arg)
