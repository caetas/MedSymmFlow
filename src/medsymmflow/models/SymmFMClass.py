##############################################################################################
############ Code based on: https://bm371613.github.io/conditional-flow-matching/ ############
### and https://github.com/openai/guided-diffusion                                         ###
##############################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import zuko
from torchvision.utils import make_grid
from config import models_dir
import os
from torchdiffeq import odeint
from diffusers.models import AutoencoderKL
from accelerate import Accelerator
import copy
import cv2
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.masks import build_palette
from .unet import UNetModel, update_ema

def create_checkpoint_dir():
    '''
    Create a directory to save the model checkpoints
    '''
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(os.path.join(models_dir, 'SymmetricalFlowMatchingClass')):
        os.makedirs(os.path.join(models_dir, 'SymmetricalFlowMatchingClass'))

class SymmFMClass(nn.Module):

    def __init__(self, args, img_size=32, in_channels=3):
        '''
        SymmetricalFlowMatchingClass module
        :param args: arguments
        :param img_size: size of the image
        :param in_channels: number of input channels
        '''
        super(SymmFMClass, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae =  AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(self.device) if args.latent else None
        self.channels = in_channels
        self.img_size = img_size
        self.rgb_mask = args.rgb_mask

        # If using VAE, change the number of channels and image size accordingly
        if self.vae is not None:
            self.channels = 4
            self.img_size = self.img_size // 8
            self.model_channels = self.channels * 2
        else:
            if self.rgb_mask:
                self.model_channels = self.channels + 3
            else:
                self.model_channels = self.channels + 1

        self.model = UNetModel(
            image_size=self.img_size,
            in_channels=self.model_channels,
            model_channels=args.model_channels,
            out_channels=self.model_channels,
            num_res_blocks=args.num_res_blocks,
            attention_resolutions=args.attention_resolutions,
            dropout=args.dropout,
            channel_mult=args.channel_mult,
            conv_resample=args.conv_resample,
            dims=args.dims,
            num_classes=None,
            use_checkpoint=False,
            num_heads=args.num_heads,
            num_head_channels=args.num_head_channels,
            num_heads_upsample=-1,
            use_scale_shift_norm=args.use_scale_shift_norm,
            resblock_updown=args.resblock_updown,
            use_new_attention_order=args.use_new_attention_order
        )

        self.model.to(self.device)
        self.lr = args.lr
        self.n_epochs = args.n_epochs
        self.sample_and_save_freq = args.sample_and_save_freq
        self.dataset = args.dataset
        self.solver = args.solver
        self.step_size = args.step_size
        self.solver_lib = args.solver_lib
        self.no_wandb = args.no_wandb
        self.warmup = args.warmup
        self.decay = args.decay
        self.snapshot = args.n_epochs//args.snapshots
        self.beta = args.beta
        self.image_weight = args.image_weight
        self.n_classes = args.n_classes
        self.palette = build_palette(4, 75)

        if args.train:
            self.ema = copy.deepcopy(self.model)
            self.ema_rate = args.ema_rate
            for param in self.ema.parameters():
                param.requires_grad = False

    def forward(self, x, t):
        '''
        Forward pass of the SymmetricalFlowMatchingClass module
        :param x: input image
        :param t: time
        '''
        return self.model(x, t)
    
    def symmetrical_flow_matching_loss(self, x, mask):
        '''
        Symmetrical flow matching loss
        :param x: input image
        :param mask: mask
        Returns:
        - Image Generation Loss, Mask Generation Loss
        '''
        sigma_min = 1e-4
        t = torch.rand(x.shape[0], device=x.device)

        noise_x = torch.randn_like(x)
        noise_mask = torch.randn_like(mask)
        x_t = (1 - (1 - sigma_min) * t[:, None, None, None]) * noise_x + t[:, None, None, None] * x
        mask_t = (1 - (1 - sigma_min) * t[:, None, None, None]) * mask + t[:, None, None, None] * noise_mask

        optimal_flow_x = x - (1 - sigma_min) * noise_x
        optimal_flow_mask = noise_mask - (1 - sigma_min) * mask
        
        input = torch.cat([x_t, mask_t], dim=1)
        
        optimal_flow = torch.cat([optimal_flow_x, optimal_flow_mask], dim=1)
        predicted_flow = self.forward(input, t)

        return (predicted_flow[:, :x.shape[1]] - optimal_flow[:, :x.shape[1]]).square().mean(), (predicted_flow[:, x.shape[1]:] - optimal_flow[:, x.shape[1]:]).square().mean()
    
    @torch.no_grad()
    def encode(self, x):
        '''
        Encode the input image
        :param x: input image
        '''
        # check if it is a distributted model or not
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.vae.module.encode(x)
        else:
            return self.vae.encode(x)

    @torch.no_grad()    
    def decode(self, z):
        '''
        Decode the input image
        :param z: input image
        '''
        # check if it is a distributted model or not
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.vae.module.decode(z)
        else:
            return self.vae.decode(z)
    
    @torch.no_grad()
    def sample(self, n_samples, mask, train=True, accelerate=None, fid=False):
        '''
        Sample images
        :param n_samples: number of samples
        :param mask: mask
        :param train: if True, sample during training
        :param accelerate: Accelerator object
        :param fid: if True, return the samples
        '''
        x_0 = torch.randn(n_samples, self.channels, self.img_size, self.img_size, device=self.device)
        x_0 = torch.cat([x_0, mask], dim=1)

        if train:
            def f(t: float, x):
                return self.ema(x, torch.full(x.shape[:1], t, device=self.device))
        else:
            def f(t: float, x):
                return self.forward(x, torch.full(x.shape[:1], t, device=self.device))
        
        if self.solver_lib == 'torchdiffeq':
            if self.solver == 'euler' or self.solver == 'rk4' or self.solver == 'midpoint' or self.solver == 'explicit_adams' or self.solver == 'implicit_adams':
                samples = odeint(f, x_0, t=torch.linspace(0, 1, 2).to(self.device), options={'step_size': self.step_size}, method=self.solver, rtol=1e-5, atol=1e-5)
            else:
                samples = odeint(f, x_0, t=torch.linspace(0, 1, 2).to(self.device), method=self.solver, options={'max_num_steps': 1//self.step_size}, rtol=1e-5, atol=1e-5)
            samples = samples[1]
        elif self.solver_lib == 'zuko':
            samples = zuko.utils.odeint(f, x_0, 0, 1, phi=self.model.parameters(), atol=1e-5, rtol=1e-5)
        else:
            t=0
            for i in tqdm(range(int(1/self.step_size)), desc='Sampling', leave=False):
                if train:
                    v = self.ema(x_0, torch.full(x_0.shape[:1], t, device=self.device))
                else:
                    v = self.forward(x_0, torch.full(x_0.shape[:1], t, device=self.device))
                x_0 = x_0 + self.step_size * v
                t += self.step_size
            samples = x_0

        samples = samples[:, :self.channels]
        
        if self.vae is not None:
            samples = self.decode(samples / 0.18215).sample
            mask = self.decode(mask / 0.18215).sample

        samples = samples*0.5 + 0.5
        samples = samples.clamp(0, 1)
        mask = mask*0.5 + 0.5
        mask = mask.clamp(0, 1)

        if fid:
            return samples
        
        fig = plt.figure(figsize=(20, 10))
        grid_mask = make_grid(mask, nrow=int(n_samples**0.5), padding=0)
        grid = make_grid(samples, nrow=int(n_samples**0.5), padding=0)
        plt.subplot(1, 2, 1)
        plt.imshow(grid_mask.permute(1, 2, 0).cpu().detach().numpy())
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())
        plt.axis('off')

        if train:
            if not self.no_wandb:
                accelerate.log({"samples": fig})
        else:
            plt.show()

        plt.close(fig)

    @torch.no_grad()
    def segment(self, n_samples, x, train=True, accelerate=None, eval=False):
        '''
        Segment images
        :param n_samples: number of samples
        :param x: input image that we want to segment
        :param train: if True, sample during training
        :param accelerate: Accelerator object
        '''
        #x_0 = torch.randn(n_samples, 1, self.img_size, self.img_size, device=self.device)
        if self.vae is not None:
            x_0 = torch.randn(n_samples, self.channels, self.img_size, self.img_size, device=self.device)
        else:
            if self.rgb_mask:
                x_0 = torch.randn(n_samples, 3, self.img_size, self.img_size, device=self.device)
            else:
                x_0 = torch.randn(n_samples, 1, self.img_size, self.img_size, device=self.device)
        x_0 = torch.cat([x, x_0], dim=1)

        if train:
            def f(t: float, x):
                return self.ema(x, torch.full(x.shape[:1], t, device=self.device))
        else:
            def f(t: float, x):
                return self.forward(x, torch.full(x.shape[:1], t, device=self.device))
        
        if self.solver_lib == 'torchdiffeq':
            if self.solver == 'euler' or self.solver == 'rk4' or self.solver == 'midpoint' or self.solver == 'explicit_adams' or self.solver == 'implicit_adams':
                samples = odeint(f, x_0, t=torch.linspace(1, 0, 2).to(self.device), options={'step_size': self.step_size}, method=self.solver, rtol=1e-5, atol=1e-5)
            else:
                samples = odeint(f, x_0, t=torch.linspace(1, 0, 2).to(self.device), method=self.solver, options={'max_num_steps': 1//self.step_size}, rtol=1e-5, atol=1e-5)
            samples = samples[1]
        elif self.solver_lib == 'zuko':
            samples = zuko.utils.odeint(f, x_0, 1, 0, phi=self.model.parameters(), atol=1e-5, rtol=1e-5)
        else:
            t=1
            for i in tqdm(range(int(1/self.step_size)), desc='Sampling', leave=False):
                if train:
                    v = self.ema(x_0, torch.full(x_0.shape[:1], t, device=self.device))
                else:
                    v = self.forward(x_0, torch.full(x_0.shape[:1], t, device=self.device))
                x_0 = x_0 - self.step_size * v
                t -= self.step_size
            samples = x_0

        samples = samples[:, self.channels:]
        
        if self.vae is not None:
            samples = self.decode(samples / 0.18215).sample
            x = self.decode(x / 0.18215).sample

        if eval:
            return samples

        samples = samples*0.5 + 0.5
        samples = samples.clamp(0, 1)
        x = x*0.5 + 0.5
        x = x.clamp(0, 1)

        # plot two grids side by side, one with the original image and the other with the segmented image
        fig = plt.figure(figsize=(20, 10))
        grid_x = make_grid(x, nrow=int(n_samples**0.5), padding=0)
        grid = make_grid(samples, nrow=int(n_samples**0.5), padding=0)
        plt.subplot(1, 2, 1)
        plt.imshow(grid_x.permute(1, 2, 0).cpu().detach().numpy())
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())
        plt.axis('off')
        if train:
            if not self.no_wandb:
                accelerate.log({"segmentations": fig})
                plt.close(fig)
        else:
            plt.show()

    def dequantize_class(self, label):
        '''
        Dequantize the mask
        :param mask: mask
        :param beta: beta value
        '''
        if self.rgb_mask:
            # we should spread the indices of the palette to push colors apart
            color_translation = torch.linspace(0, len(self.palette)-1, self.n_classes, device=self.device).long()
            #mask takes the color of the palette corresponding to the label
            if self.vae is not None:
                mask = torch.zeros(label.shape[0], 3, self.img_size*8, self.img_size*8, device=self.device).long()
            else:
                mask = torch.zeros(label.shape[0], 3, self.img_size, self.img_size, device=self.device).long()
            for i in range(self.n_classes):
                mask[label == i] = torch.tensor(self.palette[color_translation[i]], device=self.device).view(1, 3, 1, 1)

            # normalize to -1, 1
            mask = mask.float()/255.0
            mask = mask*2.0 - 1.0
            # add perturbation to the mask
            mask = mask + (self.beta)*(torch.rand_like(mask) - 0.5)/127.5
        else:
            interval = (self.beta/2)/(self.n_classes-1)
            #interval = 1.5/(self.n_classes-1)
            label = (label/(self.n_classes-1))*2.0 - 1.0
            mask = (self.beta/2)*torch.ones(label.shape[0], 1, self.img_size, self.img_size, device=self.device)*label[:, None, None, None]
            mask = mask + interval*(torch.rand_like(mask) - 0.5)

        return mask

    def quantize_class(self, mask):
        '''
        Quantize the mask
        :param mask: mask
        :param beta: beta value
        '''
        if self.rgb_mask:
            # we should spread the indices of the palette to push colors apart
            color_translation = torch.linspace(0, len(self.palette)-1, self.n_classes, device=self.device).long()
            #mask takes the color of the palette corresponding to the label
            distances = torch.zeros(mask.shape[0], self.n_classes, device=self.device)
            pixel_distances = torch.zeros(mask.shape[0], self.n_classes, mask.shape[2], mask.shape[3], device=self.device)
            for i in range(self.n_classes):
                ref_color = torch.tensor(self.palette[color_translation[i]], device=self.device).view(1, 3, 1, 1)
                #normalize to -1, 1
                ref_color = ref_color.float()/255.0
                ref_color = ref_color*2.0 - 1.0
                distances[:, i] = torch.norm(mask - ref_color, dim=1).mean(dim=(1, 2))
                pixel_distances[:, i] = torch.norm(mask - ref_color, dim=1)
            
            #predict the class with the minimum distance
            mean_prediction = distances.argmin(dim=1)
            # get the most likely class for each pixel, i.e., for each pixel choose the argmin
            mode_prediction = pixel_distances.argmin(dim=1)
            mode_prediction = mode_prediction.flatten(start_dim=1).mode(dim=1)[0]

            return mean_prediction, mode_prediction


        else:
            mask = mask/(self.beta/2)
            mask = mask*0.5 + 0.5

            # label each image as the mean class in the mask
            mean_prediction = mask.mean(dim=(1, 2, 3)) 
            mean_prediction *= (self.n_classes-1)
            mean_prediction = mean_prediction.round()
            mean_prediction = mean_prediction.clamp(0, self.n_classes-1)

            # label each image as the most frequent class in the mask
            mode_prediction = mask*(self.n_classes-1)
            mode_prediction = mode_prediction.round().clamp(0, self.n_classes-1)
            mode_prediction = mode_prediction.flatten(start_dim=1).mode(dim=1)[0]

            return mean_prediction, mode_prediction
    
    def distance_to_classes(self, mask):
        '''
        Compute the distance to each individual class
        :param mask: mask
        :param beta: beta value
        '''
        if self.rgb_mask:
            # we should spread the indices of the palette to push colors apart
            color_translation = torch.linspace(0, len(self.palette)-1, self.n_classes, device=self.device).long()
            #mask takes the color of the palette corresponding to the label
            distances = torch.zeros(mask.shape[0], self.n_classes, device=self.device)
            for i in range(self.n_classes):
                ref_color = torch.tensor(self.palette[color_translation[i]], device=self.device).view(1, 3, 1, 1)
                #normalize to -1, 1
                ref_color = ref_color.float()/255.0
                ref_color = ref_color*2.0 - 1.0
                dist_plot = torch.norm(mask - ref_color, dim=1)
                distances[:, i] = torch.norm(mask - ref_color, dim=1).mean(dim=(1, 2))
            return distances
        
        else:
            mask = mask/(self.beta/2)
            mask = mask*0.5 + 0.5
            mask = mask.mean(dim=(1, 2, 3))
            mask *= (self.n_classes-1)
            distances = torch.zeros(mask.shape[0], self.n_classes, device=self.device)
            for i in range(self.n_classes):
                distances[:, i] = torch.abs(mask - i)
            return distances   

    
    def train_model(self, train_loader, val_loader, verbose=True):
        '''
        Train the model
        :param train_loader: training data loader
        '''
        accelerate = Accelerator(log_with="wandb")
        if not self.no_wandb:
            accelerate.init_trackers(project_name='MedSymmFlowClass',
            config = {
                        "dataset": self.args.dataset,
                        "batch_size": self.args.batch_size,
                        "n_epochs": self.args.n_epochs,
                        "lr": self.args.lr,
                        "channels": self.channels,
                        "input_size": self.img_size,
                        'model_channels': self.args.model_channels,
                        'num_res_blocks': self.args.num_res_blocks,
                        'attention_resolutions': self.args.attention_resolutions,
                        'dropout': self.args.dropout,
                        'channel_mult': self.args.channel_mult,
                        'conv_resample': self.args.conv_resample,
                        'dims': self.args.dims,
                        'num_heads': self.args.num_heads,
                        'num_head_channels': self.args.num_head_channels,
                        'use_scale_shift_norm': self.args.use_scale_shift_norm,
                        'resblock_updown': self.args.resblock_updown,
                        'use_new_attention_order': self.args.use_new_attention_order,  
                        "ema_rate": self.args.ema_rate,
                        "warmup": self.args.warmup,
                        "latent": self.args.latent,
                        "decay": self.args.decay,
                        "size": self.args.size, 
                        "n_classes": self.args.n_classes,
                        "beta": self.args.beta,
                        "rgb_mask": self.args.rgb_mask,
                        "image_weight": self.args.image_weight,
                },
                init_kwargs={"wandb":{"name": f"MedSymmFlowClass_{self.args.dataset}"}})

        epoch_bar = tqdm(range(self.n_epochs), desc='Epochs', leave=True)
        create_checkpoint_dir()

        best_loss = float('inf')

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.n_epochs*len(train_loader), pct_start=self.warmup/self.n_epochs, anneal_strategy='cos', cycle_momentum=False, div_factor=self.lr/1e-6, final_div_factor=1)

        if  self.vae is None:
            train_loader, self.model, optimizer, scheduler, self.ema, val_loader = accelerate.prepare(train_loader, self.model, optimizer, scheduler, self.ema, val_loader)
        else:
            train_loader, self.model, optimizer, scheduler, self.ema, self.vae, val_loader = accelerate.prepare(train_loader, self.model, optimizer, scheduler, self.ema, self.vae, val_loader)


        update_ema(self.ema, self.model, 0)

        for epoch in epoch_bar:
            self.model.train()
            train_loss_image = 0.0
            train_loss_mask = 0.0
            for x, label in tqdm(train_loader, desc='Batches', leave=False, disable=not verbose):
                x = x.to(self.device)

                if label.dim() > 1:
                    label = label.squeeze(1)

                mask = self.dequantize_class(label)
                mask = mask.to(self.device)

                with accelerate.autocast():

                    if self.vae is not None:
                        with torch.no_grad():
                            # if x has one channel, make it 3 channels
                            if x.shape[1] == 1:
                                x = torch.cat((x, x, x), dim=1)
                            if mask.shape[1] == 1:
                                mask = torch.cat((mask, mask, mask), dim=1)
                            x = self.encode(x).latent_dist.sample().mul_(0.18215)
                            mask = self.encode(mask).latent_dist.mode().mul_(0.18215)

                    optimizer.zero_grad()
                    loss_image, loss_mask = self.symmetrical_flow_matching_loss(x, mask)
                    loss = self.image_weight*loss_image + (1.-self.image_weight)*loss_mask
                    accelerate.backward(loss)
                optimizer.step()
                scheduler.step()
                train_loss_image += loss_image.item()*x.size(0)
                train_loss_mask += loss_mask.item()*x.size(0)
                update_ema(self.ema, self.model, self.ema_rate)
            
            accelerate.wait_for_everyone()

            if not self.no_wandb:
                accelerate.log({"Train Loss Image": train_loss_image / len(train_loader.dataset)})
                accelerate.log({"Train Loss Mask": train_loss_mask / len(train_loader.dataset)})
                accelerate.log({"Learning Rate": scheduler.get_last_lr()[0]})

            epoch_bar.set_postfix({'Loss': (train_loss_image+train_loss_mask)*0.5 / len(train_loader.dataset)})

            if (epoch+1) % self.sample_and_save_freq == 0 or epoch == 0:
                self.model.eval()
                # one batch from the validation loader
                x, label = next(iter(val_loader))
                x = x.to(self.device)
                if label.dim() > 1:
                    label = label.squeeze(1)
                mask = self.dequantize_class(label)
                mask = mask.to(self.device)
                if self.vae is not None:
                    with torch.no_grad():
                        if x.shape[1] == 1:
                            x = torch.cat((x, x, x), dim=1)
                        if mask.shape[1] == 1:
                            mask = torch.cat((mask, mask, mask), dim=1)
                        x = self.encode(x).latent_dist.sample().mul_(0.18215)
                        mask = self.encode(mask).latent_dist.mode().mul_(0.18215)
                self.sample(x.shape[0], mask, accelerate=accelerate)
                #self.segment(x.shape[0], x, accelerate=accelerate)
            
            if (epoch+1) % self.snapshot == 0:
                ema_to_save = accelerate.unwrap_model(self.ema)
                accelerate.save(ema_to_save.state_dict(), os.path.join(models_dir, 'SymmetricalFlowMatchingClass', f"{'LatFM' if self.vae is not None else 'FM'}_{self.dataset}_beta{self.beta}{'_rgb' if self.rgb_mask else ''}_epoch{epoch+1}.pt"))

        accelerate.end_training()

    def load_checkpoint(self, checkpoint_path):
        '''
        Load a model checkpoint
        :param checkpoint_path: path to the checkpoint
        '''
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, weights_only=False))

    @torch.no_grad()
    def evaluate_segmentation(self, dataloader):
        '''
        Evaluate the segmentation
        :param dataloader: data loader
        '''
        self.model.eval()
        gt = []
        pred = []
        mode_pred = []
        distances = []
        for x, label in tqdm(dataloader, desc='Evaluating', leave=True):
            x = x.to(self.device)
            gt.append(label.numpy())

            if self.vae is not None:
                with torch.no_grad():
                    if x.shape[1] == 1:
                        x = torch.cat((x, x, x), dim=1)
                    x = self.encode(x).latent_dist.sample().mul_(0.18215)

            predicted_masks = self.segment(x.shape[0], x, train=False, eval=True)
            distances.append(self.distance_to_classes(predicted_masks).cpu().numpy())
            mean_prediction, mode_prediction = self.quantize_class(predicted_masks)
            pred.append(mean_prediction.cpu().long().numpy())
            mode_pred.append(mode_prediction.cpu().long().numpy())

        gt = np.concatenate(gt)
        pred = np.concatenate(pred)
        mode_pred = np.concatenate(mode_pred)
        distances = np.concatenate(distances)
        # get roc_auc score for each class then average them
        auc = [roc_auc_score(gt != i, distances[:,i]) for i in range(self.n_classes)]

        acc = accuracy_score(gt, pred)
        mode_acc = accuracy_score(gt, mode_pred)
        print(f"Mean Accuracy: {100*acc:.2f}%")
        print(f"Mode Accuracy: {100*mode_acc:.2f}%")
        print(f"AUC: {100*np.mean(auc):.2f}%")

        self.uncertainty_quantification(gt, pred, mode_pred, distances)

    def uncertainty_quantification(self, gt, pred, mode_pred, distances):
        '''
        Uncertainty quantification
        :param dataloader: data loader
        '''
        
        distances = distances[np.arange(distances.shape[0]), pred]

        # rank gt, pred and distances by distance
        gt = gt[np.argsort(-distances)]
        pred = pred[np.argsort(-distances)]
        mode_pred = mode_pred[np.argsort(-distances)]
        distances = distances[np.argsort(-distances)]

        # progressively eliminate the most uncertain samples at a rate of 0, then 0.1, 0.2, ..., 0.9
        # and compute the accuracy for each rate
        rates = np.arange(0, 1.0, 0.05)
        accuracies = []
        mode_accuracies = []
        for rate in rates:
            # eliminate the most uncertain samples
            n_samples = int(len(distances) * rate)
            gt_ = gt[n_samples:]
            pred_ = pred[n_samples:]
            mode_pred_ = mode_pred[n_samples:]
            acc = accuracy_score(gt_, pred_)
            accuracies.append(acc)
            mode_acc = accuracy_score(gt_, mode_pred_)
            mode_accuracies.append(mode_acc)
        
        print(accuracies)

        # plot the accuracies
        plt.plot(rates, accuracies, label='Mean Prediction')
        plt.plot(rates, mode_accuracies, label='Mode Prediction')
        plt.legend()
        plt.xlabel('Rate of most uncertain samples eliminated')
        plt.ylabel('Accuracy')
        plt.title('Uncertainty Quantification')
        plt.show()


    @torch.no_grad()
    def fid_sample(self, batch_size=16):
        '''
        Sample images for FID calculation
        :param batch_size: batch size
        '''
        # if self.args.checkpoint contains epoch number, ep = epoch number
        # else, ep = 0
        if 'epoch' in self.args.checkpoint:
            ep = int(self.args.checkpoint.split('epoch')[1].split('.')[0])
        else:
            ep = 0

        if not os.path.exists('./../../fid_samples'):
            os.makedirs('./../../fid_samples')
        if not os.path.exists(f"./../../fid_samples/{self.dataset}"):
            os.makedirs(f"./../../fid_samples/{self.dataset}")
        #add solverlib, solver, stepsize
        if not os.path.exists(f"./../../fid_samples/{self.dataset}/fm_{self.solver_lib}_solver_{self.solver}_stepsize_{self.step_size}_ep{ep}"):
            os.makedirs(f"./../../fid_samples/{self.dataset}/fm_{self.solver_lib}_solver_{self.solver}_stepsize_{self.step_size}_ep{ep}")
        cnt = 0
        labels = torch.arange(50000, device=self.device) % self.n_classes
        for i in tqdm(range(50000//batch_size), desc='FID Sampling', leave=True):
            mask = self.dequantize_class(labels[i*batch_size:(i+1)*batch_size])
            mask = mask.to(self.device)
            samps = self.sample(batch_size, mask, train=False, fid=True).cpu().numpy()
            samps = (samps*255).astype(np.uint8)
            samps = samps.transpose(0, 2, 3, 1)
            for samp in samps:
                cv2.imwrite(f"./../../fid_samples/{self.dataset}/fm_{self.solver_lib}_solver_{self.solver}_stepsize_{self.step_size}_ep{ep}/{cnt}.png", cv2.cvtColor(samp, cv2.COLOR_RGB2BGR) if samp.shape[-1] == 3 else samp)
                cnt += 1 




