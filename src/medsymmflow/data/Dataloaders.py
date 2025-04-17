from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import torch
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
from tqdm import tqdm
from torchvision import datasets
from config import data_raw_dir
import zipfile
import os
from glob import glob
from medmnist import RetinaMNIST, BloodMNIST, PneumoniaMNIST, DermaMNIST
from utils.masks import build_palette


def retinamnist_train_loader(batch_size, normalize = True, input_shape = None, num_workers = 0):
        
    if normalize:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    #3 available sizes for download: 28, 64, 128 and 224, choose the closest to input_shape
    if input_shape is not None:
        size = min([28, 64, 128, 224], key=lambda x: abs(x - input_shape))
    else:
        size = 28
    training_data = RetinaMNIST(root=data_raw_dir, split='train', download=True, transform=transform, size = size)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers = num_workers)
    
    if input_shape is not None:
        return input_shape, 3, training_loader
    else:
        return 32, 3, training_loader
        
def retinamnist_val_loader(batch_size, normalize = True, input_shape = None):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    #3 available sizes for download: 28, 64, 128 and 224, choose the closest to input_shape
    if input_shape is not None:
        size = min([28, 64, 128, 224], key=lambda x: abs(x - input_shape))
    else:
        size = 28

    validation_data = RetinaMNIST(root=data_raw_dir, split='test', download=True, transform=transform, size = size)

    validation_loader = DataLoader(validation_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True)
    
    if input_shape is not None:
        return input_shape, 3, validation_loader
    else:
        return 32, 3, validation_loader
    
def bloodmnist_train_loader(batch_size, normalize = True, input_shape = None, num_workers = 0):
        
        if normalize:
            transform = transforms.Compose([
                transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
                transforms.ToTensor(),
            ])
    
        if input_shape is not None:
            size = min([64, 128, 224], key=lambda x: abs(x - input_shape))
            training_data = BloodMNIST(root=data_raw_dir, split='train', download=True, transform=transform, size = size)
        else:
            training_data = BloodMNIST(root=data_raw_dir, split='train', download=True, transform=transform)
    
        training_loader = DataLoader(training_data, 
                                    batch_size=batch_size, 
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers = num_workers)
        
        if input_shape is not None:
            return input_shape, 3, training_loader
        else:
            return 32, 3, training_loader
        
def bloodmnist_val_loader(batch_size, normalize = True, input_shape = None):

        
            if normalize:
                transform = transforms.Compose([
                    transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
                    transforms.ToTensor(),
                ])
    
            if input_shape is not None:
                size = min([64, 128, 224], key=lambda x: abs(x - input_shape))
                validation_data = BloodMNIST(root=data_raw_dir, split='test', download=True, transform=transform, size = size)
            else:
                validation_data = BloodMNIST(root=data_raw_dir, split='test', download=True, transform=transform)
    
            validation_loader = DataLoader(validation_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True)
            
            if input_shape is not None:
                return input_shape, 3, validation_loader
            else:
                return 32, 3, validation_loader
            
def dermamnist_train_loader(batch_size, normalize = True, input_shape = None, num_workers = 0):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    if input_shape is not None:
        size = min([64, 128, 224], key=lambda x: abs(x - input_shape))
        training_data = DermaMNIST(root=data_raw_dir, split='train', download=True, transform=transform, size = size)
    else:
        training_data = DermaMNIST(root=data_raw_dir, split='train', download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers = num_workers)
    
    if input_shape is not None:
        return input_shape, 3, training_loader
    else:
        return 32, 3, training_loader
    
def dermamnist_val_loader(batch_size, normalize = True, input_shape = None):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    if input_shape is not None:
        size = min([64, 128, 224], key=lambda x: abs(x - input_shape))
        validation_data = DermaMNIST(root=data_raw_dir, split='test', download=True, transform=transform, size = size)
    else:
        validation_data = DermaMNIST(root=data_raw_dir, split='test', download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return input_shape, 3, validation_loader
    else:
        return 32, 3, validation_loader
    
def pneumoniamnist_train_loader(batch_size, normalize = True, input_shape = None, num_workers = 0):
                            
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    if input_shape is not None:
        size = min([64, 128, 224], key=lambda x: abs(x - input_shape))
        training_data = PneumoniaMNIST(root=data_raw_dir, split='train', download=True, transform=transform, size = size)
    else:
        training_data = PneumoniaMNIST(root=data_raw_dir, split='train', download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers = num_workers)

    if input_shape is not None:
        return input_shape, 1, training_loader
    else:
        return 32, 1, training_loader 

def pneumoniamnist_val_loader(batch_size, normalize = True, input_shape = None):
                                    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])
    
    if input_shape is not None:
        size = min([64, 128, 224], key=lambda x: abs(x - input_shape))
        validation_data = PneumoniaMNIST(root=data_raw_dir, split='test', download=True, transform=transform, size = size)
    else:
        validation_data = PneumoniaMNIST(root=data_raw_dir, split='test', download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return input_shape, 1, validation_loader
    else:
        return 32, 1, validation_loader


class SurgeSAMDataset(Dataset):
    def __init__(self, transform_fn, mask_transform_fn, mode='train', size=256):
        '''
        Initializes the SurgeSAMDataset
        Args:
            transform_fn: function
            mode: str
        '''
        self.images = glob(os.path.join(data_raw_dir, "SurgeSAM_processed", "*", "*", "*", "images", "*.jpg"))
        self.images.sort()
        val_videos = ['s8k98gGeFf0.mp4', 'watch#v=xEps8nqblY0.mp4']
        if mode == 'train':
            self.images = [img for img in self.images if not any(video in img for video in val_videos)]
        elif mode == 'val':
            self.images = [img for img in self.images if any(video in img for video in val_videos)]
        
        self.masks = [img.replace("images", "machine_masks").replace(".jpg", ".png") for img in self.images]
        self.transform_fn = transform_fn
        self.mask_transform_fn = mask_transform_fn
        self.size = size
        self.mode = mode
        self.palette = build_palette(4, 75)

    def __len__(self):
        '''
        Returns the length of the dataset
        Returns:
            int
        '''
        return len(self.images)
    
    def __getitem__(self, idx):
        '''
        Returns the image and mask at the given index
        Args:
            idx: int
        Returns:
            image: torch.Tensor
            mask: torch.Tensor
        '''
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')

        # Find smallest dimension
        min_dim = min(image.size)

        # Randomly choose a starting point within the allowed range for both width and height
        left = np.random.randint(0, image.size[0] - min_dim + 1) if image.size[0] > min_dim else 0
        top = np.random.randint(0, image.size[1] - min_dim + 1) if image.size[1] > min_dim else 0

        right = left + min_dim
        bottom = top + min_dim

        # Apply the random crop
        image = image.crop((left, top, right, bottom))
        mask = mask.crop((left, top, right, bottom))
        # resize the image and mask to the input_shape
        image = image.resize((self.size, self.size))
        mask = mask.resize((self.size, self.size), resample=Image.NEAREST)
        image = self.transform_fn(image)
        mask = self.mask_to_color(mask)
        mask = self.mask_transform_fn(mask)
        if self.mode == 'train':
            # flip the image and mask horizontally with a 50% chance
            if np.random.rand() > 0.5:
                image = torch.flip(image, [-1])
                mask = torch.flip(mask, [-1])
        return image, mask

    def mask_to_color(self, mask):
        mask = np.array(mask)
        mask = mask.squeeze()
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for unique in np.unique(mask):
            if unique == 255:
                unique = len(self.palette) - 1
                colored_mask[mask == 255] = self.palette[unique]
            else:
                colored_mask[mask == unique] = self.palette[unique]
        return Image.fromarray(colored_mask)
    
def surge_sam_dataloader(batch_size, num_workers, mode='train', input_shape=None):
    '''
    Returns a DataLoader for the SurgeSAMDataset
    Args:
        batch_size: int
        num_workers: int
        mode: str
        input_shape: int
    Returns:
        DataLoader
    '''
    transform = transforms.Compose([
        transforms.Resize((input_shape, input_shape)) if input_shape is not None else transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((input_shape, input_shape), interpolation=transforms.InterpolationMode.NEAREST) if input_shape is not None else transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    input_shape = input_shape if input_shape is not None else 256

    dataset = SurgeSAMDataset(transform_fn=transform, mask_transform_fn=transform_mask, mode=mode, size=input_shape)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=(mode != 'train'))

    return input_shape, 3, dataloader

def pick_dataset(name, mode='train', input_shape=None, batch_size=64, num_workers=0):
    '''
    Picks the dataset based on the name
    Args:
        name: str
        mode: str
        input_shape: int
        batch_size: int
        num_workers: int
    Returns:
        DataLoader
    '''
    if name == 'pneumoniamnist':
        if mode == 'train':
            return pneumoniamnist_train_loader(batch_size, normalize=True, input_shape=input_shape, num_workers=num_workers)
        else:
            return pneumoniamnist_val_loader(batch_size, normalize=True, input_shape=input_shape)
    elif name == 'bloodmnist':
        if mode == 'train':
            return bloodmnist_train_loader(batch_size, normalize=True, input_shape=input_shape, num_workers=num_workers)
        else:
            return bloodmnist_val_loader(batch_size, normalize=True, input_shape=input_shape)
    elif name == 'dermamnist':
        if mode == 'train':
            return dermamnist_train_loader(batch_size, normalize=True, input_shape=input_shape, num_workers=num_workers)
        else:
            return dermamnist_val_loader(batch_size, normalize=True, input_shape=input_shape)
    elif name == 'retinamnist':
        if mode == 'train':
            return retinamnist_train_loader(batch_size, normalize=True, input_shape=input_shape, num_workers=num_workers)
        else:
            return retinamnist_val_loader(batch_size, normalize=True, input_shape=input_shape)
    elif name == 'surgesam':
        return surge_sam_dataloader(batch_size, num_workers, mode=mode, input_shape=input_shape)
    else:
        raise ValueError(f"Dataset {name} not found. Available datasets: pneumoniamnist, bloodmnist, dermamnist, retinamnist, surgesam")
        