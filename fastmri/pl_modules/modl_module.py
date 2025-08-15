import torch
import torch.nn as nn
import pytorch_lightning as pl
from fastmri.data import transforms as fastmri_transforms
from fastmri.pl_modules.mri_module import MriModule
from fastmri.models.modl import MoDL
from fastmri.models.varnet import SensitivityModel
import fastmri

class MoDLModule(MriModule):
    """
    PyTorch Lightning module for MoDL model using fastMRI data pipeline.
    """
    def __init__(
        self,
        n_layers: int = 5,
        k_iters: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        sens_chans: int = 8,
        sens_pools: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = MoDL(n_layers=n_layers, k_iters=k_iters)
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            in_chans=2,
            out_chans=2,
            drop_prob=0.0,
            mask_center=True,
        )
        
        self.loss = fastmri.SSIMLoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MoDL")
        parser.add_argument("--n_layers", default=5, type=int, help="Number of CNN layers in MoDL denoiser")
        parser.add_argument("--k_iters", default=10, type=int, help="Number of MoDL iterations")
        parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay")
        parser.add_argument("--sens_chans", default=8, type=int, help="Sensitivity map U-Net channels")
        parser.add_argument("--sens_pools", default=4, type=int, help="Sensitivity map U-Net pools")
        return parent_parser

    def forward(self, image, csm, mask):
        return self.model(image, csm, mask)

    def _prepare_inputs(self, batch):
        masked_kspace, mask, target, attrs, fname, slice_num = batch
        csm = self.sens_net(masked_kspace, mask)
        image = fastmri_transforms.ifft2c(masked_kspace)
        image = fastmri_transforms.complex_abs(image)
        image = fastmri_transforms.complex_center_crop(image, (image.shape[-2], image.shape[-1]))
        image = fastmri_transforms.to_tensor(image)
        mask = mask.squeeze(1)
        target = fastmri_transforms.to_tensor(target)
        return image, csm, mask, target

    def training_step(self, batch, batch_idx):
        image, csm, mask, target = self._prepare_inputs(batch)
        output = self(image, csm, mask)
        loss = self.loss(output, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, csm, mask, target = self._prepare_inputs(batch)
        output = self(image, csm, mask)
        val_loss = self.loss(output, target)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        image, csm, mask, target = self._prepare_inputs(batch)
        output = self(image, csm, mask)
        test_loss = self.loss(output, target)
        self.log("test_loss", test_loss)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
