import torch
import torch.nn as nn

# Utility functions r2c, c2r should be implemented or imported from fastMRI utils if available
# For now, we define simple versions here

def r2c(x, axis=1):
    # Convert real tensor (B, 2, H, W) to complex (B, H, W)
    return torch.complex(x[:, 0], x[:, 1])

def c2r(x, axis=1):
    # Convert complex tensor (B, H, W) to real (B, 2, H, W)
    return torch.stack([x.real, x.imag], dim=axis)

# CNN denoiser ======================
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class cnn_denoiser(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        layers = []
        layers += conv_block(2, 64)
        for _ in range(n_layers-2):
            layers += conv_block(64, 64)
        layers += nn.Sequential(
            nn.Conv2d(64, 2, 3, padding=1),
            nn.BatchNorm2d(2)
        )
        self.nw = nn.Sequential(*layers)
    def forward(self, x):
        idt = x
        dw = self.nw(x) + idt
        return dw

# CG algorithm ======================
class myAtA(nn.Module):
    def __init__(self, csm, mask, lam):
        super(myAtA, self).__init__()
        self.csm = csm
        self.mask = mask
        self.lam = lam
    def forward(self, im):
        im_coil = self.csm * im
        k_full = torch.fft.fft2(im_coil, norm='ortho')
        k_u = k_full * self.mask
        im_u_coil = torch.fft.ifft2(k_u, norm='ortho')
        im_u = torch.sum(im_u_coil * self.csm.conj(), axis=1)
        return im_u + self.lam * im

def myCG(AtA, rhs):
    rhs = r2c(rhs, axis=1)
    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rTr = torch.sum(r.conj()*r).real
    while i < 10 and rTr > 1e-10:
        Ap = AtA(p)
        alpha = rTr / torch.sum(p.conj()*Ap).real
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(r.conj()*r).real
        beta = rTrNew / rTr
        p = r + beta * p
        i += 1
        rTr = rTrNew
    return c2r(x, axis=1)

class data_consistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)
    def forward(self, z_k, x0, csm, mask):
        rhs = x0 + self.lam * z_k
        AtA = myAtA(csm, mask, self.lam)
        rec = myCG(AtA, rhs)
        return rec

# MoDL model =======================    
class MoDL(nn.Module):
    def __init__(self, n_layers, k_iters):
        super().__init__()
        self.k_iters = k_iters
        self.dw = cnn_denoiser(n_layers)
        self.dc = data_consistency()
    def forward(self, x0, csm, mask):
        x_k = x0.clone()
        for k in range(self.k_iters):
            z_k = self.dw(x_k)
            x_k = self.dc(z_k, x0, csm, mask)
        return x_k
