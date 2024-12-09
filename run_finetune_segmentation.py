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
from utils.misc_update import NativeScalerWithGradNormCount as NativeScaler
import swin_unet
from torchvision.transforms import functional as TF
import utils_f
from utils.engine_finetune import train_one_epoch, evaluate, DiceLoss
from datasets import build_pretraining_dataset
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
import random
import yacs
import json
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
    
def elastic_transform(image, alpha, sigma, random_state=None, is_mask=False):
    """Apply elastic transformation on an image.

    Args:
        image (numpy.ndarray): Input image.
        alpha (float): Scaling factor for the transformations.
        sigma (float): Gaussian filter's standard deviation.
        random_state (np.random.RandomState, optional): Random state for reproducibility.
        mode (str, optional): The mode parameter determines how the input array is extended beyond its boundaries. 
                               Default is 'reflect'. Use 'nearest' for masks.

    Returns:
        numpy.ndarray: Transformed image.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    if image.ndim == 3 and image.shape[2] == 3:  # Check if the image is RGB
        # Assuming all channels are the same, just use the first channel
        single_channel = image[:, :, 0]
        
        shape = single_channel.shape
        dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        transformed_channel = ndimage.map_coordinates(single_channel, indices, order=1, mode='reflect').reshape(shape)

        # Replicate the transformed channel across all three channels
        transformed_image = np.stack([transformed_channel] * 3, axis=-1)
        return transformed_image
    else:
        shape = image.shape
        dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        if is_mask:
            return ndimage.map_coordinates(image, indices, order=0, mode='nearest').reshape(shape)
        else:
            return ndimage.map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


    
class JointTransform:
    def __init__(self, size,dir_name,apply_transform):
        self.size = size  # Size can be a tuple (width, height) or an integer
        self.do_transform = apply_transform
        self.affine_transform = transforms.RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
        self.dir_name = dir_name
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
            if random.random() > 0.5:
                random_state = np.random.RandomState(None)
                angle, translations, scale, shear = self.affine_transform.get_params(self.affine_transform.degrees, self.affine_transform.translate, self.affine_transform.scale, self.affine_transform.shear, self.size)
                image = TF.affine(Image.fromarray(image), angle, translations, scale, shear)
                mask = TF.affine(Image.fromarray(mask), angle, translations, scale, shear, interpolation=TF.InterpolationMode.NEAREST)
                image = np.asarray(image)
                mask = np.asarray(mask)

        # Convert the thresholded mask back to a tensor
        image= Image.fromarray(image)
        mask = Image.fromarray(mask)
        return image, mask  
        
def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # common parameters
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--transform', action='store_true')
    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--gpu_num', default=1, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--nnunet',  action='store_true')
    parser.add_argument('--datapercent', default='', type=str)
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
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')

    # other parameters
    parser.add_argument('--task', default='amos', type=str)
    parser.add_argument('--fold_num', default='0', type=str)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    #parser.add_argument('--log_dir', default='./output_dir',
    #                    help='path where to tensorboard log')
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
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser

class SegmentationDataset(Dataset):
    def __init__(self,dataper, root_dir, image_size, fold_num, split="train",add_transform=True,use_nnunet=True, use_simclr=False):
        self.root_dir = root_dir   ### the csv path containing image list and mask list
        if dataper!='':
            self.dataframe = pd.read_csv(os.path.join(root_dir,dataper+'_'+split+'_fold'+fold_num+'.csv'))
        else:
            self.dataframe = pd.read_csv(os.path.join(root_dir,split+'_fold'+fold_num+'.csv'))
        self.split = split
        self.joint_transform = JointTransform(size=(image_size, image_size),dir_name = root_dir,apply_transform=add_transform)  # Example size
        self.use_nnunet = use_nnunet
        #no transform for segmentation task. This could be any additional image-specific transform
        self.use_simclr = use_simclr
        self.images = self.dataframe['image']
        self.masks = self.dataframe['mask']
        
            
        self.to_tensor = transforms.ToTensor() 
        #self.images = sorted(os.listdir(self.images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
            img_name = self.images[idx]
            mask_name = self.masks[idx]
    
            image = Image.open(img_name).convert("RGB")
            mask = Image.open(mask_name).convert("L")  # Assuming mask is grayscale
            
            ## for [0,255]
            #mask_ = np.array(mask) / 255.0
            #mask_ = (mask_ > 0.5).astype(np.float32)
            #mask = Image.fromarray(mask_)
            # Apply joint resize transform
            #image = np.array(image)
            #mask = np.array(mask)
            image, mask = self.joint_transform(image, mask)
            image_arr = np.asarray(image)
            if self.use_simclr==False:
                image_arr = (image_arr - image_arr.min()+1e-6)/(image_arr.max()-image_arr.min()+1e-6)
            image = torch.tensor(image_arr, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(np.asarray(mask))
            #image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            #mask = torch.from_numpy(mask.astype(np.float32))
            mask = mask.long()
            
            #print(image.shape,mask.shape)
            #image = TF.to_tensor(image)
            #mask = TF.to_tensor(mask).squeeze(0).long() 
            #print(image.shape,mask.shape)
            return image, mask, img_name
        
        
def main(args):
    if args.distributed:
        utils_f.init_distributed_mode(args)
    
        print(args)
    
        device = torch.device(args.device)
    
        # fix the seed for reproducibility
        seed = args.seed + utils_f.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        # random.seed(seed)
    
        cudnn.benchmark = True
    else:
        seed = args.seed + utils_f.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device(args.device)
        
    # Set model
    model = swin_unet.__dict__['swin_unet'](num_classes=args.num_classes)
    
    # get dataset
    transform_train = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Set dataset
    if 'simclr' in args.finetune:
        use_simclr=True
    else:
        use_simclr=False
    #dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    dataset_train = SegmentationDataset(dataper = args.datapercent,root_dir=args.data_path,image_size = args.input_size, fold_num = args.fold_num,split="train" ,add_transform=args.transform, use_nnunet = args.nnunet,use_simclr=use_simclr)
    if args.eval == True:
        dataset_val = SegmentationDataset(dataper = '',root_dir=args.data_path,image_size = args.input_size,fold_num = args.fold_num, split="test" ,add_transform=False, use_nnunet = args.nnunet,use_simclr=use_simclr)
    else:
        dataset_val = SegmentationDataset(dataper = '',root_dir=args.data_path,image_size = args.input_size,fold_num = args.fold_num, split="val" ,add_transform=False, use_nnunet = False,use_simclr=use_simclr)
    if args.distributed:  # args.distributed:
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
    
    status = 'pretrain'
    
    ### create output file name
    study_name = args.output_dir  +status
    
    if args.eval==True:
        study_name = args.output_dir +'/'+args.finetune.split('/')[-2]
    if args.distributed==True and global_rank == 0:
        os.makedirs(study_name, exist_ok=True)
    
    
    if args.eval!=True:
        with open(os.path.join(study_name, "running_param.txt"), 'w') as file:
            json.dump(str(args), file)
    if args.distributed==True and global_rank == 0 and study_name is not None:
        os.makedirs(study_name, exist_ok=True)
        log_writer = SummaryWriter(logdir=study_name)
        #log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    elif args.distributed!=True and study_name is not None:
        os.makedirs(study_name, exist_ok=True)
        log_writer = SummaryWriter(logdir=study_name)
    else:
        log_writer = None
    
    model.to(device)
    model_without_ddp = model

    # Set optimizer
    param_groups = [p for p in model_without_ddp.parameters() if p.requires_grad]
    
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=5e-2, betas=(0.9, 0.95))  # ???5E-2
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=0.0001)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=args.lr, eps= 1e-06, weight_decay=0.1)
    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    # Create model
    misc.load_model(args=args, model_without_ddp=model_without_ddp)    
    
    max_dice = 0.0
    
    
    ### test evaluation
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, True,study_name, dice_loss)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['dice_score']:.3f}")
        exit(0)
    
    # Start the training process
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, dice_loss, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        #if args.output_dir and ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
        #    misc.save_model(
        #        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #        loss_scaler=loss_scaler, epoch=epoch + 1)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }
        
        
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device,False,'',dice_loss)
            print(f"Dice score of the network on the {len(dataset_val)} val images: {test_stats['dice_score']:.3f}")
            if max_dice < test_stats["dice_score"]:
                max_dice = test_stats["dice_score"]
                
                misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler,study_name=study_name, epoch="best")

            print(f'Max dice: {max_dice:.3f}')


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch}
        
        if study_name and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(study_name, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    arg = get_args_parser()
    arg = arg.parse_args()

    status = 'pretrain'
    study_name = arg.output_dir +status
    
    if arg.eval==True:
        study_name =arg.output_dir +'/'+ arg.finetune.split('/')[-2]
    
    if arg.output_dir:
        Path(study_name).mkdir(parents=True, exist_ok=True)
    main(arg)
