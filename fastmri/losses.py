"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        reduced: bool = True,
    ):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if reduced:
            return 1 - S.mean()
        else:
            return 1 - S

class MS_SSIMLoss(nn.Module):
    def __init__(
        self,
        size_average=True,
        win_size: int = 7,
        win_sigma=1.5,
        channel=1,
        spatial_dims=2,
        weights=None,
        K=(0.01, 0.03),
    ):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIMLoss, self).__init__()
        self.win_size = win_size
        self.win = self._fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.weights = weights
        self.K = K
        
    def _fspecial_gauss_1d(self, size, sigma):
        r"""Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (1 x 1 x size)
        """
        coords = torch.arange(size, dtype=torch.float)
        coords -= size // 2

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        return g.unsqueeze(0).unsqueeze(0)
    
    def gaussian_filter(self, input, win):
        r""" Blur input with 1-D kernel
        Args:
            input (torch.Tensor): a batch of tensors to be blurred
            window (torch.Tensor): 1-D gauss kernel

        Returns:
            torch.Tensor: blurred tensors
        """
        assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
        if len(input.shape) == 4:
            conv = F.conv2d
        elif len(input.shape) == 5:
            conv = F.conv3d
        else:
            raise NotImplementedError(input.shape)

        C = input.shape[1]
        out = input
        for i, s in enumerate(input.shape[2:]):
            if s >= win.shape[-1]:
                out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
            else:
                print(
                    f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
                )

        return out
    
    def _ssim(self, X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):

        r""" Calculate ssim index for X and Y

        Args:
            X (torch.Tensor): images
            Y (torch.Tensor): images
            win (torch.Tensor): 1-D gauss kernel
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

        Returns:
            torch.Tensor: ssim results.
        """
        K1, K2 = K
        # batch, channel, [depth,] height, width = X.shape
        compensation = 1.0

        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2

        win = win.to(X.device, dtype=X.dtype)

        mu1 = self.gaussian_filter(X, win)
        mu2 = self.gaussian_filter(Y, win)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = compensation * (self.gaussian_filter(X * X, win) - mu1_sq)
        sigma2_sq = compensation * (self.gaussian_filter(Y * Y, win) - mu2_sq)
        sigma12 = compensation * (self.gaussian_filter(X * Y, win) - mu1_mu2)

        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
        ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

        ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
        cs = torch.flatten(cs_map, 2).mean(-1)
        return ssim_per_channel, cs
    
    def ms_ssim(
        self, X, Y, data_range=1.0, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)
    ):

        r""" interface of ms-ssim
        Args:
            X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
            Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        Returns:
            torch.Tensor: ms-ssim results
        """
        if not X.shape == Y.shape:
            raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

        for d in range(len(X.shape) - 1, 1, -1):
            X = X.squeeze(dim=d)
            Y = Y.squeeze(dim=d)

        if not X.type() == Y.type():
            raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

        if len(X.shape) == 4:
            avg_pool = F.avg_pool2d
        elif len(X.shape) == 5:
            avg_pool = F.avg_pool3d
        else:
            raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

        if win is not None:  # set win_size
            win_size = win.shape[-1]

        if not (win_size % 2 == 1):
            raise ValueError("Window size should be odd.")

        smaller_side = min(X.shape[-2:])
        assert smaller_side > (win_size - 1) * (
            2 ** 4
        ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = X.new_tensor(weights)

        if win is None:
            win = self._fspecial_gauss_1d(win_size, win_sigma)
            win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

        levels = weights.shape[0]
        mcs = []
        for i in range(levels):
            ssim_per_channel, cs = self._ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

            if i < levels - 1:
                mcs.append(torch.relu(cs))
                padding = [s % 2 for s in X.shape[2:]]
                X = avg_pool(X, kernel_size=2, padding=padding)
                Y = avg_pool(Y, kernel_size=2, padding=padding)

        ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
        mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
        ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

        if size_average:
            return 1 - ms_ssim_val.mean()
        else:
            return 1 - ms_ssim_val.mean(1)
    

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        data_range = data_range[:, None, None, None]
        return self.ms_ssim(
            X,
            Y,
            data_range=data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor, feature_layers=[2], style_layers=[0,1,2,3]):
        if X.shape[1] != 3:
            X = X.repeat(1, 3, 1, 1)
            Y = Y.repeat(1, 3, 1, 1)
        X = (X-self.mean) / self.std
        Y = (Y-self.mean) / self.std
        if self.resize:
            X = self.transform(X, mode='bilinear', size=(224, 224), align_corners=False)
            Y = self.transform(Y, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = X
        y = Y
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class AlexNetPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(AlexNetPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.alexnet(pretrained=True).features[:2].eval())
        blocks.append(torchvision.models.alexnet(pretrained=True).features[2:5].eval())
        blocks.append(torchvision.models.alexnet(pretrained=True).features[5:8].eval())
        blocks.append(torchvision.models.alexnet(pretrained=True).features[8:10].eval())
        blocks.append(torchvision.models.alexnet(pretrained=True).features[10:12].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor, feature_layers=[2], style_layers=[0,1,2,3,4]):
        if X.shape[1] != 3:
            X = X.repeat(1, 3, 1, 1)
            Y = Y.repeat(1, 3, 1, 1)
        X = (X-self.mean) / self.std
        Y = (Y-self.mean) / self.std
        if self.resize:
            X = self.transform(X, mode='bilinear', size=(224, 224), align_corners=False)
            Y = self.transform(Y, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = X
        y = Y
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss     

LAPLACIAN_FILTER = np.array([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]],
                             dtype=np.float32)

class NLPDLoss(nn.Module):
    """
    Normalised lapalcian pyramid distance.
    Refer to https://www.cns.nyu.edu/pub/eero/laparra16a-preprint.pdf
    https://github.com/alexhepburn/nlpd-tensorflow
    """
    def __init__(self, channels=1, k=6, filt=None):
        super(NLPDLoss, self).__init__()
        if filt is None:
            filt = np.reshape(np.tile(LAPLACIAN_FILTER, (channels, 1, 1)),
                              (channels, 1, 5, 5))
        self.k = k
        self.channels = channels
        self.filt = nn.Parameter(torch.Tensor(filt), requires_grad=False)
        self.dn_filts, self.sigmas = self.DN_filters()
        self.pad_one = nn.ReflectionPad2d(1)
        self.pad_two = nn.ReflectionPad2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def DN_filters(self):
        sigmas = [0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782]
        dn_filts = []
        dn_filts.append(torch.Tensor(np.reshape([[0, 0.1011, 0],
                                    [0.1493, 0, 0.1460],
                                    [0, 0.1015, 0.]]*self.channels,
                                   (self.channels,	1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0.0757, 0],
                                    [0.1986, 0, 0.1846],
                                    [0, 0.0837, 0]]*self.channels,
                                   (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0.0477, 0],
                                    [0.2138, 0, 0.2243],
                                    [0, 0.0467, 0]]*self.channels,
                                   (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
                                    [0.2503, 0, 0.2616],
                                    [0, 0, 0]]*self.channels,
                                   (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
                                    [0.2598, 0, 0.2552],
                                    [0, 0, 0]]*self.channels,
                                   (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
                                    [0.2215, 0, 0.0717],
                                    [0, 0, 0]]*self.channels,
                                   (self.channels, 1, 3, 3)).astype(np.float32)))
        dn_filts = nn.ParameterList([nn.Parameter(x, requires_grad=False)
                                     for x in dn_filts])
        sigmas = nn.ParameterList([nn.Parameter(torch.Tensor(np.array(x)),
                                  requires_grad=False) for x in sigmas])
        return dn_filts, sigmas

    def pyramid(self, im):
        out = []
        J = im
        pyr = []
        for i in range(0, self.k):
            I = F.conv2d(self.pad_two(J), self.filt, stride=2, padding=0,
                         groups=self.channels)
            I_up = self.upsample(I)
            I_up_conv = F.conv2d(self.pad_two(I_up), self.filt, stride=1,
                                 padding=0, groups=self.channels)
            if J.size() != I_up_conv.size():
                I_up_conv = F.interpolate(I_up_conv, [J.size(2), J.size(3)])
            out = J - I_up_conv
            out_conv = F.conv2d(self.pad_one(torch.abs(out)), self.dn_filts[i],
                         stride=1, groups=self.channels)
            out_norm = out / (self.sigmas[i]+out_conv)
            pyr.append(out_norm)
            J = I
        return pyr

    def nlpd(self, x1, x2):
        y1 = self.pyramid(x1)
        y2 = self.pyramid(x2)           
        total = []
        for z1, z2 in zip(y1, y2):
            diff = (z1 - z2) ** 2
            sqrt = torch.sqrt(torch.mean(diff, (1, 2, 3)))
            total.append(sqrt)
        score = torch.stack(total,dim=1).mean(1)
        return score

    def forward(self, y, x, data_range, as_loss=True):
        assert x.shape == y.shape
        if as_loss:
            score = self.nlpd(x, y)
            return score.mean()
        else:
            with torch.no_grad():
                score = self.nlpd(x, y)
            return score