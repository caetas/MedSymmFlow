from data.Dataloaders import *
from utils.util import parse_args_SymmetricFlowMatchingClass
from models.SymmFMClass import SymmFMClass
import torch

if __name__ == '__main__':
    args = parse_args_SymmetricFlowMatchingClass()

    if args.train:
        
        image_shape, channels, dataloader = pick_dataset(args.dataset, 'train', args.size, args.batch_size, args.num_workers)
        _, _, dataloader_val = pick_dataset(args.dataset, 'val', args.size, args.batch_size, args.num_workers)

        model = SymmFMClass(args, image_shape, channels)
        #model.sample(16, mask=mask, train=False)
        #model.segment(16, x, train=False)
        model.train_model(dataloader, dataloader_val)

    elif args.sample:
        
        image_shape, channels, dataloader = pick_dataset(args.dataset, 'val', args.size, args.batch_size, args.num_workers)

        model = SymmFMClass(args, image_shape, channels)
        model.load_checkpoint(args.checkpoint)

        # create 16 masks for the 16 samples, based on the classes
        labels = torch.arange(0, args.num_samples).to(model.device) % args.n_classes
        mask = model.dequantize_class(labels)
        mask = mask.to(model.device)
        model.sample(args.num_samples, mask=mask, train=False)

    elif args.classification:
        
        image_shape, channels, dataloader = pick_dataset(args.dataset, 'val', args.size, args.batch_size, args.num_workers)
        model = SymmFMClass(args, image_shape, channels)
        model.load_checkpoint(args.checkpoint)
        model.evaluate_segmentation(dataloader)

    else:
        
        image_shape, channels, dataloader = pick_dataset(args.dataset, 'val', args.size, args.batch_size, args.num_workers)

        model = SymmFMClass(args, image_shape, channels)
        model.load_checkpoint(args.checkpoint)
        model.fid_sample(args.batch_size)