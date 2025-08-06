import torch.nn as nn

from models.deformer.rigid import get_rigid_deform
from models.deformer.non_rigid import get_non_rigid_deform

class Deformer(nn.Module):
    def __init__(self, cfg, metadata, subjects=None, multigaussian=False):
        super().__init__()
        self.cfg = cfg
        self.rigid = get_rigid_deform(cfg.rigid, metadata, subjects=subjects, multigaussian=multigaussian)
        self.non_rigid = get_non_rigid_deform(cfg.non_rigid, metadata, subjects=subjects, multigaussian=multigaussian)

    def forward(self, gaussians, camera, iteration, compute_loss=True):
        loss_reg = {}
        deformed_gaussians, loss_non_rigid = self.non_rigid(gaussians, iteration, camera, compute_loss)
        deformed_gaussians = self.rigid(deformed_gaussians, iteration, camera)

        loss_reg.update(loss_non_rigid)
        return deformed_gaussians, loss_reg

def get_deformer(cfg, metadata, subjects=None, multigaussian=False):
    return Deformer(cfg, metadata, subjects=subjects, multigaussian=multigaussian)