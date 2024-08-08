import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import cv2
import pywt
import pywt.data
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
        a =  np.random.choice(1, 1, replace=False)
        '''
        arr = cv2.imread("/home/mhduan/projectsummer/duannat/DIV2K_part/DIV2K_valid_HR/0803.png")
        arr = cv2.resize(arr, (512, 512))
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        image = pywt.swt2(arr, 'bior1.3', level=1, start_level=0)
        image = np.array(image)
        foo = image[0][0]
        image_normorlize_L = (foo - foo.min()) / (foo.max() - foo.min())
        print(image_normorlize_L,"image_normorlize_L")
        foo = image[0][1][0]
        image_normorlize_H0 = (foo - foo.min()) / (foo.max() - foo.min())
        foo = image[0][1][1]
        image_normorlize_H1 = (foo - foo.min()) / (foo.max() - foo.min())
        '''
        arr = pywt.data.aero()

        #plt.imshow(arr, interpolation="nearest", cmap=plt.cm.gray)
        #plt.imshow(arr)
        #plt.show()


        level = 0
        titles = ['Approximation', ' Horizontal detail',
                'Vertical detail', 'Diagonal detail']
        pywtswt2 = pywt.swt2(arr, 'haar', level=1, start_level=0)
        for LL, (LH, HL, HH) in pywt.swt2(arr, 'haar', level=1, start_level=0):
            fig = plt.figure()
            for i, a in enumerate([LL, LH, HL, HH]):
                ax = fig.add_subplot(2, 2, i + 1)
                ax.imshow(a, origin='upper', interpolation="nearest", cmap=plt.cm.gray)
                ax.set_title(titles[i], fontsize=12)

            fig.suptitle("SWT2 coefficients, level %s" % level, fontsize=14)
            level += 1


        plt.show()



    #LL, LH, HL, HH = get_wav(in_channels=3)
