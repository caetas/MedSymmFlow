import argparse

def parse_args_SymmetricFlowMatching():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample model')
    argparser.add_argument('--fid', action='store_true', default=False, help='calculate FID')
    argparser.add_argument('--batch_size', type=int, default=256, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--model_channels', type=int, default = 64, help='number of features')
    argparser.add_argument('--num_res_blocks', type=int, default = 2, help='number of residual blocks per downsample')
    argparser.add_argument('--attention_resolutions', type=int, nargs='+', default = [4], help='downsample rates at which attention will take place')
    argparser.add_argument('--dropout', type=float, default = 0.0, help='dropout probability')
    argparser.add_argument('--channel_mult', type=int, nargs='+', default = [1, 2, 2], help='channel multiplier for each level of the UNet')
    argparser.add_argument('--conv_resample', type=bool, default = True, help='use learned convolutions for upsampling and downsampling')
    argparser.add_argument('--dims', type=int, default = 2, help='determines if the signal is 1D, 2D, or 3D')
    argparser.add_argument('--num_heads', type=int, default = 4, help='number of attention heads in each attention layer')
    argparser.add_argument('--num_head_channels', type=int, default = 32, help='use a fixed channel width per attention head')
    argparser.add_argument('--use_scale_shift_norm', type=bool, default = False, help='use a FiLM-like conditioning mechanism')
    argparser.add_argument('--resblock_updown', type=bool, default = False, help='use residual blocks for up/downsampling')
    argparser.add_argument('--use_new_attention_order', type=bool, default = False, help='use a different attention pattern for potentially increased efficiency')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample and save frequency')
    argparser.add_argument('--dataset', type=str, default='celeba', help='dataset name', choices=['celeba', 'cityscapes', 'coco', 'ade20k'])
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--num_samples', type=int, default=16, help='number of samples')
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'tinyimagenet','imagenet'])
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--interpolation', action='store_true', default=False, help='interpolation')
    argparser.add_argument('--solver_lib', type=str, default='none', help='solver library', choices=['torchdiffeq', 'zuko', 'none'])
    argparser.add_argument('--step_size', type=float, default=0.1, help='step size for ODE solver')
    argparser.add_argument('--solver', type=str, default='dopri5', help='solver for ODE', choices=['dopri5', 'rk4', 'dopri8', 'euler', 'bosh3', 'adaptive_heun', 'midpoint', 'explicit_adams', 'implicit_adams'])
    argparser.add_argument('--no_wandb', action='store_true', default=False, help='disable wandb logging')
    argparser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
    argparser.add_argument('--warmup', type=int, default=10, help='warmup epochs')
    argparser.add_argument('--decay', type=float, default=1e-7, help='decay rate')
    argparser.add_argument('--latent', action='store_true', default=False, help='Use latent implementation')
    argparser.add_argument('--size', type=int, default=None, help='Size of the original image')
    argparser.add_argument('--ema_rate', type=float, default=0.999, help='ema rate')
    argparser.add_argument('--snapshots', type=int, default=10, help='how many snapshots during training')
    argparser.add_argument('--beta', type=float, default=15, help='Dequantization factor for the mask')
    argparser.add_argument('--image_weight', type=float, default=.9, help='Weight for the image loss')
    argparser.add_argument('--eval', action='store_true', default=False, help='evaluate model')
    args = argparser.parse_args()
    args.channel_mult = tuple(args.channel_mult)
    args.attention_resolutions = tuple(args.attention_resolutions)
    return args

def parse_args_SymmetricFlowMatchingClass():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample model')
    argparser.add_argument('--fid', action='store_true', default=False, help='calculate FID')
    argparser.add_argument('--batch_size', type=int, default=256, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--model_channels', type=int, default = 64, help='number of features')
    argparser.add_argument('--num_res_blocks', type=int, default = 2, help='number of residual blocks per downsample')
    argparser.add_argument('--attention_resolutions', type=int, nargs='+', default = [4], help='downsample rates at which attention will take place')
    argparser.add_argument('--dropout', type=float, default = 0.0, help='dropout probability')
    argparser.add_argument('--channel_mult', type=int, nargs='+', default = [1, 2, 2], help='channel multiplier for each level of the UNet')
    argparser.add_argument('--conv_resample', type=bool, default = True, help='use learned convolutions for upsampling and downsampling')
    argparser.add_argument('--dims', type=int, default = 2, help='determines if the signal is 1D, 2D, or 3D')
    argparser.add_argument('--num_heads', type=int, default = 4, help='number of attention heads in each attention layer')
    argparser.add_argument('--num_head_channels', type=int, default = 32, help='use a fixed channel width per attention head')
    argparser.add_argument('--use_scale_shift_norm', type=bool, default = False, help='use a FiLM-like conditioning mechanism')
    argparser.add_argument('--resblock_updown', type=bool, default = False, help='use residual blocks for up/downsampling')
    argparser.add_argument('--use_new_attention_order', type=bool, default = False, help='use a different attention pattern for potentially increased efficiency')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample and save frequency')
    argparser.add_argument('--dataset', type=str, default='pneumoniamnist', help='dataset name', choices=['pneumoniamnist', 'bloodmnist', 'dermamnist', 'retinamnist'])
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--num_samples', type=int, default=16, help='number of samples')
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--interpolation', action='store_true', default=False, help='interpolation')
    argparser.add_argument('--solver_lib', type=str, default='none', help='solver library', choices=['torchdiffeq', 'zuko', 'none'])
    argparser.add_argument('--step_size', type=float, default=0.1, help='step size for ODE solver')
    argparser.add_argument('--solver', type=str, default='dopri5', help='solver for ODE', choices=['dopri5', 'rk4', 'dopri8', 'euler', 'bosh3', 'adaptive_heun', 'midpoint', 'explicit_adams', 'implicit_adams'])
    argparser.add_argument('--no_wandb', action='store_true', default=False, help='disable wandb logging')
    argparser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
    argparser.add_argument('--warmup', type=int, default=10, help='warmup epochs')
    argparser.add_argument('--decay', type=float, default=1e-7, help='decay rate')
    argparser.add_argument('--latent', action='store_true', default=False, help='Use latent implementation')
    argparser.add_argument('--size', type=int, default=None, help='Size of the original image')
    argparser.add_argument('--ema_rate', type=float, default=0.999, help='ema rate')
    argparser.add_argument('--snapshots', type=int, default=10, help='how many snapshots during training')
    argparser.add_argument('--beta', type=float, default=15, help='Dequantization factor for the mask')
    argparser.add_argument('--image_weight', type=float, default=.9, help='Weight for the image loss')
    argparser.add_argument('--n_classes', type=int, default=10, help='Number of classes')
    argparser.add_argument('--classification', action='store_true', default=False, help='evaluate model')
    argparser.add_argument('--rgb_mask', action='store_true', default=False, help='use rgb mask')
    args = argparser.parse_args()
    args.channel_mult = tuple(args.channel_mult)
    args.attention_resolutions = tuple(args.attention_resolutions)
    return args

# EOF
