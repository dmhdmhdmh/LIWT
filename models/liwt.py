import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from models import register
from utils import make_coord
from models.SearchTransfer import SearchTransfer
import numpy as np
# Discrete Wavelet Transform
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
             kernel_size=1, stride=1, padding=0)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=1, padding=0)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=1, padding=0)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0)

    return LL, LH, HL, HH

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

# Wavelet High-Frequency Enhancement Residual Block
class HFERB(nn.Module):
    def __init__(
        self, n_feats, act=nn.ReLU(True)):
        super(HFERB, self).__init__()
        LFE = []
        HFE = []
        LFE.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        LFE.append(act)
        HFE.append(nn.MaxPool2d(3, stride=1, padding=1))
        HFE.append(nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0))
        HFE.append(act)
        self.dim = n_feats
        self.LFE = nn.Sequential(*LFE)
        self.HFE = nn.Sequential(*HFE)
        self.convout = nn.Conv2d(n_feats*2, n_feats, kernel_size=1, padding=0)

    def forward(self, x):
        xlfe = self.LFE(x)
        xhfe = self.HFE(x)
        xcat = torch.cat([xlfe,xhfe],dim=1)
        xout = x + self.convout(xcat)
        return xout

# wavelet enhancement residual block  
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True)):

        super(ResBlock, self).__init__()
        ml = []
        for i in range(1):
            ml.append(HFERB(n_feats, act=nn.ReLU(True)))
            if bn:
                ml.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                ml.append(act)
        self.bodyl = nn.Sequential(*ml)
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)        
        self.convout = conv(2*n_feats, n_feats, kernel_size, bias=bias)

    def forward(self, x, xl):
        resl = self.bodyl(xl)
        res = self.body(x)
        rescat = torch.cat([res,resl],dim=1)
        resout = self.convout(rescat)
        resx = x + resout
        resxl = xl + resout

        return resx, resxl

# Wavelet Enhancement Residual Module
class Reswaveattention(nn.Module):
    def __init__(self, conv=default_conv):
        super(Reswaveattention, self).__init__()
        n_feats = 64
        kernel_size = 3
        act = nn.ReLU(True)

        m_head = [conv(3 * n_feats, n_feats, kernel_size)]
        m_headl = [conv(1 * n_feats, n_feats, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.headl = nn.Sequential(*m_headl)
        self.body1 =  ResBlock(conv, n_feats, kernel_size, act=act)
        self.body2 =  ResBlock(conv, n_feats, kernel_size, act=act)
        self.body3 =  ResBlock(conv, n_feats, kernel_size, act=act)
        self.body4 =  ResBlock(conv, n_feats, kernel_size, act=act)        

        self.convout = conv(2*n_feats, n_feats, kernel_size, bias=True)
        self.out_dim = n_feats

    def forward(self, x, xl):
        x = self.head(x)
        xl = self.headl(xl)
        resx, resxl = self.body1(x,xl)
        resx, resxl = self.body2(resx,resxl)
        resx, resxl = self.body3(resx,resxl)
        resx, resxl = self.body4(resx,resxl)
        rescat = torch.cat([resx,resxl],dim=1)
        res = self.convout(rescat)
        x = res + x

        return x

# Local Implicit Wavelet Transformer
@register('liwt')
class LIWT(nn.Module):
    def __init__(self, encoder_spec, pb_spec=None, imnet_spec=None, base_dim=192, head=8, r=1, hidden_dim=256):
        super().__init__()        
        self.encoder = models.make(encoder_spec)

        self.dim = base_dim
        self.head = head
        self.r = r
        self.r_dw = r

        self.encoder_wave = Reswaveattention(conv=default_conv) 
        self.LL, self.LH, self.HL, self.HH = get_wav(self.dim // 4)
  
        self.wave_Q = nn.Conv2d(self.dim // 4, self.dim, kernel_size=1, padding=0)  

        self.conv_k = nn.Conv2d(self.dim // 4, self.dim, kernel_size=1, padding=0)
        self.conv_v = nn.Conv2d(self.dim // 4, self.dim, kernel_size=1, padding=0)

        self.conv_head = nn.Conv2d(self.dim // 2, self.dim // 4, kernel_size=1, padding=0)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(self.dim // 2, self.dim // 4)
        self.fc_2 = nn.Linear(self.dim // 4, self.dim // 2)

        self.pb_encoder = models.make(pb_spec, args={'head': self.head}).cuda()

        self.r_area = (2 * self.r + 1)**2

        self.r_dw_area = (2 * self.r_dw + 1)**2

        self.imnet_in_dim = self.dim * self.r_area + 2 

        self.proj = nn.Linear(self.dim + self.dim // 4, self.dim)

        self.imnet = models.make(imnet_spec, args={'in_dim': self.imnet_in_dim}) 
       
        self.imnet_wave = models.make(imnet_spec, args={'in_dim': self.imnet_in_dim})

    def gen_feat(self, inp):
        self.inp = inp

        self.feat = self.encoder(inp)

        self.feat_LL = self.LL(self.feat)
        self.feat_LH = self.LH(self.feat)
        self.feat_HL = self.HL(self.feat)
        self.feat_HH = self.HH(self.feat)
        self.feat_H = torch.cat([self.feat_LH,self.feat_HL,self.feat_HH],dim=1)

        self.feat_wave = self.encoder_wave(self.feat_H, self.feat_LL)
 
        self.feat_K = self.conv_k(self.feat)
        self.feat_V = self.conv_v(self.feat)

        lr_h = self.feat.shape[-2]
        lr_w = self.feat.shape[-1]
        self.feat_wave_up = F.interpolate(self.feat_wave, size=(lr_h,lr_w), mode='bicubic')

        self.feat_wave_Q = self.wave_Q(self.feat_wave_up)

        # Wavelet Mutual Projected Fusion
        feat_cat = torch.cat([self.feat_wave_up, self.feat], dim=1)
        residual = feat_cat
        feat_pooling = self.pooling(feat_cat)
        N, C, _, _ = feat_pooling.size()
        feat_pooling = feat_pooling.view(N, -1, C)
        feat_fc_1 = self.fc_1(feat_pooling)
        feat_fc_1 = nn.ReLU(inplace=True)(feat_fc_1)
        feat_fc_2 = self.fc_2(feat_fc_1)
        weight = nn.Sigmoid()(feat_fc_2)
        weight = weight.view(N, C, 1, 1)
        out = residual
        out = out * weight
        self.cross_feat = self.conv_head(out + residual)

        return self.feat

    def query_rgb(self, coord, cell=None, wave_cell=None):
        feat = self.feat
        feat_wave = self.feat_wave_up

        bs, q_sample, _ = coord.shape

        coord_lr = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1). \
                              unsqueeze(0).expand(bs, 2, *feat.shape[-2:])

        # b, q, 1, 2
        sample_coord_ = coord.clone()
        sample_coord_ = sample_coord_.unsqueeze(2)

        # field radius (global: [-1, 1])
        rh = 2 / feat.shape[-2]
        rw = 2 / feat.shape[-1]

        r = self.r

        # b, 2, h, w -> b, 2, q, 1 -> b, q, 1, 2
        sample_coord_k = F.grid_sample(
            coord_lr, sample_coord_.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)

        dh = torch.linspace(-r, r, 2 * r + 1).cuda() * rh
        dw = torch.linspace(-r, r, 2 * r + 1).cuda() * rw
        # 1, 1, r_area, 2
        delta = torch.stack(torch.meshgrid(dh, dw, indexing='ij'), axis=-1).view(1, 1, -1, 2)

        sample_feat_waveQ = F.grid_sample(
            self.feat_wave_Q, sample_coord_.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)       

        sample_feat_waveQ = sample_feat_waveQ.reshape(
            bs, q_sample, 1, self.head, self.dim // self.head
        ).permute(0, 1, 3, 2, 4)

        # b, q, 1, 2 -> b, q, 9, 2
        sample_coord_k = sample_coord_k + delta

        # b, q, 1, 2 -> b, q, 9, 2

        # K - b, c, h, w -> b, c, q, 9 -> b, q, 9, c -> b, q, 9, h, c -> b, q, h, c, 9
        sample_feat_K = F.grid_sample(
            self.feat_K, sample_coord_k.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)
        sample_feat_K = sample_feat_K.reshape(
            bs, q_sample, self.r_area, self.head, self.dim // self.head
        ).permute(0, 1, 3, 4, 2)

        sample_feat_V = F.grid_sample(
            self.feat_V, sample_coord_k.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)

        sample_cross_feat_V = F.grid_sample(
            self.cross_feat, sample_coord_k.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)       

        # b, q, 9, 2
        rel_coord = sample_coord_ - sample_coord_k
        rel_coord[..., 0] *= feat.shape[-2]
        rel_coord[..., 1] *= feat.shape[-1]

        # b, 2 -> b, q, 2
        rel_cell = cell.clone()
        rel_cell = rel_cell.unsqueeze(1).repeat(1, q_sample, 1)
        rel_cell[..., 0] *= feat.shape[-2]
        rel_cell[..., 1] *= feat.shape[-1]

        # b, q, 9, 2
        # Wavelet-aware Implicit Attention
        attn = torch.matmul(sample_feat_waveQ, sample_feat_K).reshape(
            bs, q_sample, self.head, self.r_area
        ).permute(0, 1, 3, 2) / np.sqrt(self.dim // self.head)

        # position encoding
        _, pb = self.pb_encoder(rel_coord)
        attn = F.softmax(torch.add(attn, pb), dim=-2)


        attn = attn.reshape(bs, q_sample, self.r_area, self.head, 1)

        feat_cat = torch.cat((sample_feat_V, sample_cross_feat_V), dim=-1)

        sample_feat_V = sample_feat_V + self.proj(feat_cat)

        sample_feat_V = sample_feat_V.reshape(
            bs, q_sample, self.r_area, self.head, self.dim // self.head
        )

        sample_feat_V = torch.mul(sample_feat_V, attn).reshape(bs, q_sample, self.r_area, -1)

        feat_in = sample_feat_V.reshape(bs, q_sample, -1)

        feat_in = torch.cat([feat_in, rel_cell], dim=-1)

        pred_in = self.imnet(feat_in)

        pred = pred_in + F.grid_sample(self.inp, sample_coord_.flip(-1), mode='bilinear',\
                                       padding_mode='border', align_corners=False)[:, :, :, 0].permute(0, 2, 1)

        return pred

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
