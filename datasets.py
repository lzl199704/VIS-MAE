# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch
from torchvision.transforms import functional as TF
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from timm.data import create_transform

#from masking_generator import RandomMaskingGenerator
from dataset_folder import ImageFolder
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def get_unique_classes(masks_dir):
    unique_classes = set()
    for mask_file in os.listdir(masks_dir)[:10]:
        mask_path = os.path.join(masks_dir, mask_file)
        mask = Image.open(mask_path)
        unique_classes |= set(np.unique(mask))
    return unique_classes

class JointTransform:
    def __init__(self, size):
        self.size = size  # Size can be a tuple (width, height) or an integer

    def __call__(self, image, mask):
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
    
        # Convert the thresholded mask back to a tensor
        return image, mask    
    
class DataAugmentationForMAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.transform = transforms.Compose([
            #transforms.Resize(args.input_size),
            transforms.RandomResizedCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size, args.mask_ratio
        )

    def __call__(self, image):
        return self.transform(image), self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, image_size, split="train"):
        self.root_dir = root_dir
        self.split = split
        self.joint_transform = JointTransform(size=(image_size, image_size))  # Example size
        #no transform for segmentation task. This could be any additional image-specific transform
        self.images_dir = os.path.join(root_dir, split, "img")
        self.masks_dir = os.path.join(root_dir, split, "msk")
        self.to_tensor = transforms.ToTensor() 
        #self.images = sorted(os.listdir(self.images_dir))
        self.images = sorted([img for img in os.listdir(self.images_dir) if not img.startswith(".")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.images[idx])
        mask_name = os.path.join(self.masks_dir, self.images[idx])

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")  # Assuming mask is grayscale
        
        ## for [0,255]
        #mask_ = np.array(mask) / 255.0
        #mask_ = (mask_ > 0.5).astype(np.float32)
        #mask = Image.fromarray(mask_)
        # Apply joint resize transform
        
        image, mask = self.joint_transform(image, mask)
        image = torch.tensor(np.asarray(image), dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(np.asarray(mask))
        mask = mask.long()
        
        #print(image.shape,mask.shape)
        #image = TF.to_tensor(image)
        #mask = TF.to_tensor(mask).squeeze(0).long() 
        #print(image.shape,mask.shape)
        return image, mask, img_name

def build_pretraining_dataset(args):
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))
    return ImageFolder(args.data_path, transform=transform)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_dataset_seg(is_train, args):
    
    
        # Assuming the dataset directory structure is: root/train/image, root/train/mask, etc.
    if args.eval:
        dataset = SegmentationDataset(root_dir=args.data_path,image_size = args.input_size, split="test")
    else:
        dataset = SegmentationDataset(root_dir=args.data_path,image_size = args.input_size, split="train" if is_train else "val")

    # Calculating unique classes in the training dataset's segmentation maps
    if is_train:
        masks_dir = os.path.join(args.data_path, "train", "msk")
        unique_classes = get_unique_classes(masks_dir)
        num_classes = len(unique_classes)
    else:
        num_classes = None  # For validation and test sets, we may not need to calculate this

    return dataset, num_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
