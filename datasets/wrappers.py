import functools
import random
import math
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
from datasets import register
from utils import to_pixel_samples
from utils import make_coord
#import pywt

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0)\
        .expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0)\
        .expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0)\
        .expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0)\
        .expand(in_channels, -1, -1, -1).clone()

    return LL, LH, HL, HH


# Learnable wavelet
def get_learned_wav(in_channels, pool=True):

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0)

    return LL, LH, HL, HH


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, batch_size=1, window_size=0, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
        self.window_size = window_size
        self.batch_size = batch_size
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels=1)

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, datas):
        coords = []
        hr_list = []
        lr_list = []
        lr_h_list = []     
        lr_l_list = [] 

        scale = datas[0]['img_hr'].shape[-2] // datas[0]['img_lr'].shape[-2]

        if self.inp_size is None:
            # batch_size: 1          

            for idx, data in enumerate(datas):
                lr_h = data['img_lr'].shape[-2]
                lr_w = data['img_lr'].shape[-1]     
                img_h = data['img_hr'].shape[-2]
                img_w = data['img_hr'].shape[-1]                   
                img_wave_h = img_h//2 
                img_wave_w = img_w//2                     

                lr_gray = (255 * data['img_lr']).permute(1,2,0).numpy().astype(np.uint8)
  
                lr_gray = cv2.cvtColor(lr_gray, cv2.COLOR_RGB2GRAY)

                lr_gray = torch.from_numpy(lr_gray)
  
                lr_gray = lr_gray.unsqueeze(0).unsqueeze(0).to(torch.float32) / 255

                lr_ll = self.LL(lr_gray)
                lr_hl = self.HL(lr_gray)
                lr_lh = self.LH(lr_gray)
                lr_hh = self.HH(lr_gray)


                lr_l = lr_ll
                lr_h = torch.cat([lr_hl, lr_lh, lr_hh], dim=1)

                hr_list.append(data['img_hr'])
                lr_list.append(data['img_lr'])
                #hr_h_list.append(crop_hr_h.squeeze(0))
                lr_h_list.append(lr_h.squeeze(0))  
                lr_l_list.append(lr_l.squeeze(0))

        else:
            lr_h = self.inp_size
            lr_w = self.inp_size

            img_h = round(lr_h * scale)
            img_w = round(lr_w * scale)

            img_wave_h = img_h//2 
            img_wave_w = img_w//2 

            coords = make_coord((img_h, img_w), flatten=False).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

            for idx, data in enumerate(datas):
                h0 = random.randint(0, data['img_lr'].shape[-2] - self.inp_size)
                w0 = random.randint(0, data['img_lr'].shape[-1] - self.inp_size)
                crop_lr = data['img_lr'][:, h0:h0 + self.inp_size, w0:w0 + self.inp_size]
                hr_size = self.inp_size * scale
                h1 = h0 * scale
                w1 = w0 * scale
                crop_hr = data['img_hr'][:, h1:h1 + hr_size, w1:w1 + hr_size]

                crop_lr_gray = (255 * crop_lr).permute(1,2,0).numpy().astype(np.uint8)
    
                crop_lr_gray = cv2.cvtColor(crop_lr_gray, cv2.COLOR_RGB2GRAY)

                crop_lr_gray = torch.from_numpy(crop_lr_gray)

                crop_lr_gray = crop_lr_gray.unsqueeze(0).unsqueeze(0).to(torch.float32) / 255

                crop_lr_ll = self.LL(crop_lr_gray)
                crop_lr_hl = self.HL(crop_lr_gray)
                crop_lr_lh = self.LH(crop_lr_gray)
                crop_lr_hh = self.HH(crop_lr_gray)

                crop_lr_l = crop_lr_ll
                crop_lr_h = torch.cat([crop_lr_hl, crop_lr_lh, crop_lr_hh], dim=1)
                #crop_hr_h = torch.cat([crop_hr_hl, crop_hr_lh, crop_hr_hh], dim=1)
                #hr_coord_samples, hr_rgb_samples = to_pixel_samples(crop_hr.contiguous())
                #wave_hr_h_coord_samples, wave_hr_rgb_samples = to_pixel_samples(crop_hr_h.contiguous())  

                hr_list.append(crop_hr)
                lr_list.append(crop_lr)
                #hr_h_list.append(crop_hr_h.squeeze(0))
                lr_h_list.append(crop_lr_h.squeeze(0))                
                lr_l_list.append(crop_lr_l.squeeze(0))

        coords = make_coord((img_h, img_w), flatten=False) \
                                .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        #lr_list = [resize_fn(hr_list[i], (lr_h, lr_w)) for i in range(len(hr_list))]
        inp = torch.stack(lr_list, dim=0)
        hr_rgb = torch.stack(hr_list, dim=0)
        wave_inp = torch.stack(lr_h_list, dim=0)
        wave_inp_l = torch.stack(lr_l_list, dim=0)
        #wave_hr_rgb = torch.stack(hr_h_list, dim=0)       

        cell = torch.ones(2)
        cell[0] *= 2. / img_h
        cell[1] *= 2. / img_w
        cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

        wave_cell = torch.ones(2)
        wave_cell[0] *= 2. / img_wave_h
        wave_cell[1] *= 2. / img_wave_w
        wave_cell = wave_cell.unsqueeze(0).repeat(self.batch_size, 1)       

        if self.inp_size is None and self.window_size != 0:
            # SwinIR Evaluation - reflection padding
            # batch size : 1 for testing
            h_old, w_old = inp.shape[-2:]
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old

            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[..., :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[..., :w_old + w_pad]

            lr_h += h_pad
            lr_w += w_pad

            img_h = round(lr_h * scale)
            img_w = round(lr_w * scale)

            img_wave_h = img_h//2 
            img_wave_w = img_w//2

            coords = make_coord((img_h, img_w), flatten=False) \
                            .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)


            cell = torch.ones(2)
            cell[0] *= 2. / img_h
            cell[1] *= 2. / img_w
            cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

            wave_cell = torch.ones(2)
            wave_cell[0] *= 2. / img_wave_h
            wave_cell[1] *= 2. / img_wave_w
            wave_cell = wave_cell.unsqueeze(0).repeat(self.batch_size, 1)            

        if self.sample_q is None:
            sample_coord = None
        else:
            sample_coord = []       
            for i in range(len(hr_list)):
                flatten_coord = coords[i].reshape(-1, 2)
                sample_list = np.random.choice(flatten_coord.shape[0], self.sample_q, replace=False)
                sample_flatten_coord = flatten_coord[sample_list, :]

                sample_coord.append(sample_flatten_coord)

            sample_coord = torch.stack(sample_coord, dim=0)

        return {'inp': inp, 'gt': hr_rgb, 'coords': coords, 'cell': cell, 'sample_coord': sample_coord,
                'wave_lr': wave_inp, 'wave_lr_l': wave_inp_l, 'wave_cell': wave_cell}   

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            img_lr = augment(img_lr)
            img_hr = augment(img_hr)

        return {'img_lr': img_lr, 'img_hr': img_hr}  
    


@register('sr-implicit-paired-fast')
class SRImplicitPairedFast(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            h_hr = s * h_lr
            w_hr = s * w_lr
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        
        if self.inp_size is not None:
            x0 = random.randint(0, h_hr - h_lr)
            y0 = random.randint(0, w_hr - w_lr)
            
            hr_coord = hr_coord[x0:x0+self.inp_size, y0:y0+self.inp_size, :]
            hr_rgb = crop_hr[:, x0:x0+self.inp_size, y0:y0+self.inp_size]
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
    
    
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):
    
    def __init__(self, dataset, inp_size=None, batch_size=32, scale_min=1, scale_max=None, window_size=0,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.batch_size = batch_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.scale_max_s6 = 6
        self.scale_max_s8 = 8
        self.scale_max_s12 = 12
        #self.scale_max_s18 = 18
        self.augment = augment
        self.window_size = window_size
        self.sample_q = sample_q

        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels=1)

    def collate_fn(self, datas):
        coords = []
        hr_list = []
        lr_list = []

        scale = random.uniform(self.scale_min, self.scale_max)

        coords_s6 = []
        hr_list_s6 = []
        lr_list_s6 = []

        scale_s6 = random.uniform(self.scale_min, self.scale_max_s6)

        coords_s8 = []
        hr_list_s8 = []
        lr_list_s8 = []

        scale_s8 = random.uniform(self.scale_min, self.scale_max_s8)

        if self.inp_size is None:
            # batch_size: 1
            lr_h = math.floor(datas[0]['inp'].shape[-2] / scale + 1e-9)
            lr_w = math.floor(datas[0]['inp'].shape[-1] / scale + 1e-9)

            img_h = round(lr_h * scale)
            img_w = round(lr_w * scale)

            coords = make_coord((img_h, img_w), flatten=False) \
                                .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

            for idx, data in enumerate(datas):
                crop_hr = data['inp'][:, :img_h, :img_w]
                crop_lr = resize_fn(crop_hr, (lr_h, lr_w))

                hr_list.append(crop_hr)
                lr_list.append(crop_lr)
        else:
            lr_h = self.inp_size
            lr_w = self.inp_size

            img_size_min = 9999
            
            for idx, data in enumerate(datas):
                img_size_min = min(img_size_min, data['inp'].shape[-2], data['inp'].shape[-1])

            img_h = round(lr_h * scale)
            img_w = round(lr_w * scale)

            coords = make_coord((img_h, img_w), flatten=False).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

            for idx, data in enumerate(datas):
                h0 = random.randint(0, data['inp'].shape[-2] - img_h)
                w0 = random.randint(0, data['inp'].shape[-1] - img_w)
                crop_hr = data['inp'][:, h0:h0 + img_h, w0:w0 + img_w]
                crop_lr = resize_fn(crop_hr, (lr_h, lr_w))

                hr_list.append(crop_hr)
                lr_list.append(crop_lr)
 

            img_h_s6 = round(lr_h * scale_s6)
            img_w_s6 = round(lr_w * scale_s6)

            coords_s6 = make_coord((img_h_s6, img_w_s6), flatten=False).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

            for idx, data in enumerate(datas):
                h0_s6 = random.randint(0, data['inp'].shape[-2] - img_h_s6)
                w0_s6 = random.randint(0, data['inp'].shape[-1] - img_w_s6)
                crop_hr_s6 = data['inp'][:, h0_s6:h0_s6 + img_h_s6, w0_s6:w0_s6 + img_w_s6]
                crop_lr_s6 = resize_fn(crop_hr_s6, (lr_h, lr_w))

                hr_list_s6.append(crop_hr_s6)
                lr_list_s6.append(crop_lr_s6)

            img_h_s8 = round(lr_h * scale_s8)
            img_w_s8 = round(lr_w * scale_s8)

            coords_s8 = make_coord((img_h_s8, img_w_s8), flatten=False).unsqueeze(0).repeat(self.batch_size, 1, 1, 1)

            for idx, data in enumerate(datas):
                h0_s8 = random.randint(0, data['inp'].shape[-2] - img_h_s8)
                w0_s8 = random.randint(0, data['inp'].shape[-1] - img_w_s8)
                crop_hr_s8 = data['inp'][:, h0_s8:h0_s8 + img_h_s8, w0_s8:w0_s8 + img_w_s8]
                crop_lr_s8 = resize_fn(crop_hr_s8, (lr_h, lr_w))

                hr_list_s8.append(crop_hr_s8)
                lr_list_s8.append(crop_lr_s8)

        inp = torch.stack(lr_list, dim=0)
        hr_rgb = torch.stack(hr_list, dim=0)
   
        inp_s6 = torch.stack(lr_list_s6, dim=0)
        hr_rgb_s6 = torch.stack(hr_list_s6, dim=0)

        inp_s8 = torch.stack(lr_list_s8, dim=0)
        hr_rgb_s8 = torch.stack(hr_list_s8, dim=0)

        cell = torch.ones(2)
        cell[0] *= 2. / img_h
        cell[1] *= 2. / img_w
        cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

        cell_s6 = torch.ones(2)
        cell_s6[0] *= 2. / img_h_s6
        cell_s6[1] *= 2. / img_w_s6
        cell_s6 = cell_s6.unsqueeze(0).repeat(self.batch_size, 1)

        cell_s8 = torch.ones(2)
        cell_s8[0] *= 2. / img_h_s8
        cell_s8[1] *= 2. / img_w_s8
        cell_s8 = cell_s8.unsqueeze(0).repeat(self.batch_size, 1)

        if self.inp_size is None and self.window_size != 0:
            # SwinIR Evaluation - reflection padding
            # batch size : 1 for testing
            h_old, w_old = inp.shape[-2:]
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old

            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[..., :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[..., :w_old + w_pad]

            lr_h += h_pad
            lr_w += w_pad

            img_h = round(lr_h * scale)
            img_w = round(lr_w * scale)

            coords = make_coord((img_h, img_w), flatten=False) \
                            .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)


            cell = torch.ones(2)
            cell[0] *= 2. / img_h
            cell[1] *= 2. / img_w
            cell = cell.unsqueeze(0).repeat(self.batch_size, 1)   

        if self.sample_q is None:
            sample_coord = None
        else:
            sample_coord = []       
            for i in range(len(hr_list)):
                flatten_coord = coords[i].reshape(-1, 2)
                sample_list = np.random.choice(flatten_coord.shape[0], self.sample_q, replace=False)
                sample_flatten_coord = flatten_coord[sample_list, :]

                sample_coord.append(sample_flatten_coord)

            sample_coord = torch.stack(sample_coord, dim=0)

            sample_coord_s6 = []       
            for i in range(len(hr_list_s6)):
                flatten_coord_s6 = coords_s6[i].reshape(-1, 2)
                sample_list_s6 = np.random.choice(flatten_coord_s6.shape[0], self.sample_q, replace=False)
                sample_flatten_coord_s6 = flatten_coord_s6[sample_list_s6, :]

                sample_coord_s6.append(sample_flatten_coord_s6)

            sample_coord_s6 = torch.stack(sample_coord_s6, dim=0)

            sample_coord_s8 = []       
            for i in range(len(hr_list_s8)):
                flatten_coord_s8 = coords_s8[i].reshape(-1, 2)
                sample_list_s8 = np.random.choice(flatten_coord_s8.shape[0], self.sample_q, replace=False)
                sample_flatten_coord_s8 = flatten_coord_s8[sample_list_s8, :]

                sample_coord_s8.append(sample_flatten_coord_s8)

            sample_coord_s8 = torch.stack(sample_coord_s8, dim=0)

        return {'inp': inp, 'gt': hr_rgb, 'coords': coords, 'cell': cell, 'sample_coord': sample_coord, 
                'inp_s6': inp_s6, 'gt_s6': hr_rgb_s6, 'coords_s6': coords_s6, 'cell_s6': cell_s6, 'sample_coord_s6': sample_coord_s6, 
                'inp_s8': inp_s8, 'gt_s8': hr_rgb_s8, 'coords_s8': coords_s8, 'cell_s8': cell_s8, 'sample_coord_s8': sample_coord_s8
                }

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img = self.dataset[idx]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            img = augment(img)

        return {'inp': img}   

@register('sr-implicit-downsampled-fast')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        
        if self.inp_size is not None:
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(crop_hr.shape[0], h_lr, w_lr)
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }    
    
    
@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
