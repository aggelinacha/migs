import torch
import torch.nn as nn
import pytorch3d.transforms as tf

from models.network_utils import (HierarchicalPoseEncoder,
                                  VanillaCondMLP,
                                  HannwCondMLP,
                                  HashGrid)
from utils.general_utils import quaternion_multiply

class NonRigidDeform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        raise NotImplementedError

class Identity(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        return gaussians, {}

class MLP(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder)
        d_cond = self.pose_encoder.n_output_dims

        # add latent code
        self.latent_dim = cfg.get('latent_dim', 0)
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)
        if self.identity_dim > 0:
            d_cond += self.identity_dim
            self.identity = nn.Embedding(len(metadata['subjects']), self.identity_dim)

        d_in = 3
        d_out = 3 + 3 + 4
        self.feature_dim = cfg.get('feature_dim', 0)
        d_out += self.feature_dim

        # output dimension: position + scale + rotation
        self.mlp = VanillaCondMLP(d_in, d_cond, d_out, cfg.mlp)
        self.aabb = metadata['aabb']

        self.delay = cfg.get('delay', 0)


    def forward(self, gaussians, iteration, camera, compute_loss=True):
        if iteration < self.delay:
            deformed_gaussians = gaussians.clone()
            if self.feature_dim > 0:
                setattr(deformed_gaussians, "non_rigid_feature", torch.zeros(gaussians.get_xyz.shape[0], self.feature_dim).cuda())
            return deformed_gaussians, {}

        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs)

        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(pose_feat.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(pose_feat.shape[0], -1)
            pose_feat = torch.cat([pose_feat, latent_code], dim=1)
        if self.identity_dim > 0:
            _identity_idx = camera.identity_idx
            identity_idx = torch.Tensor([_identity_idx]).long().to(pose_feat.device)
            identity_code = self.identity(identity_idx)
            identity_code = identity_code.expand(pose_feat.shape[0], -1)
            pose_feat = torch.cat([pose_feat, identity_code], dim=1)

        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        deltas = self.mlp(xyz_norm, cond=pose_feat)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]

        deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        scale_offset = self.cfg.get('scale_offset', 'logit')
        if scale_offset == 'logit':
            deformed_gaussians._scaling = gaussians._scaling + delta_scale
        elif scale_offset == 'exp':
            deformed_gaussians._scaling = torch.log(torch.clamp_min(gaussians.get_scaling + delta_scale, 1e-6))
        elif scale_offset == 'zero':
            delta_scale = torch.zeros_like(delta_scale)
            deformed_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        rot_offset = self.cfg.get('rot_offset', 'add')
        if rot_offset == 'add':
            deformed_gaussians._rotation = gaussians._rotation + delta_rot
        elif rot_offset == 'mult':
            q1 = delta_rot
            q1[0] = 1. # [1,0,0,0] represents identity rotation
            delta_rot = delta_rot[1:]
            q2 = gaussians._rotation
            # deformed_gaussians._rotation = quaternion_multiply(q1, q2)
            deformed_gaussians._rotation = tf.quaternion_multiply(q1, q2)
        else:
            raise ValueError

        if self.feature_dim > 0:
            setattr(deformed_gaussians, "non_rigid_feature", deltas[:, 10:])

        if compute_loss:
            # regularization
            loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
            loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
            loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            loss_reg = {
                'nr_xyz': loss_xyz,
                'nr_scale': loss_scale,
                'nr_rot': loss_rot
            }
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg


class HannwMLP(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder)
        # output dimension: position + scale + rotation
        self.mlp = HannwCondMLP(3, self.pose_encoder.n_output_dims, 3 + 3 + 4, cfg.mlp, dim_coord=3)
        self.aabb = metadata['aabb']


    def forward(self, gaussians, iteration, camera, compute_loss=True):
        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs)

        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        deltas = self.mlp(xyz_norm, iteration, cond=pose_feat)

        if iteration < self.cfg.mlp.embedder.kick_in_iter:
            deltas = deltas * torch.zeros_like(deltas)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, -4:]

        deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        scale_offset = self.cfg.get('scale_offset', 'logit')
        if scale_offset == 'logit':
            deformed_gaussians._scaling = gaussians._scaling + delta_scale
        elif scale_offset == 'exp':
            deformed_gaussians._scaling = torch.log(torch.clamp_min(gaussians.get_scaling + delta_scale, 1e-6))
        elif scale_offset == 'zero':
            delta_scale = torch.zeros_like(delta_scale)
            deformed_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        rot_offset = self.cfg.get('rot_offset', 'add')
        if rot_offset == 'add':
            deformed_gaussians._rotation = gaussians._rotation + delta_rot
        elif rot_offset == 'mult':
            q1 = delta_rot
            q1[0] = 1.  # [1,0,0,0] represents identity rotation
            delta_rot = delta_rot[1:]
            q2 = gaussians._rotation
            deformed_gaussians._rotation = quaternion_multiply(q1, q2)
        else:
            raise ValueError

        if compute_loss:
            # regularization
            loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
            loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
            loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            loss_reg = {
                'nr_xyz': loss_xyz,
                'nr_scale': loss_scale,
                'nr_rot': loss_rot
            }
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg

class HashGridwithMLP(NonRigidDeform):
    def __init__(self, cfg, metadata, subjects=None, multigaussian=False):
        super().__init__(cfg)
        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder)
        d_cond = self.pose_encoder.n_output_dims
        self.subjects = subjects
        self.multigaussian = multigaussian
        if "subjects" in metadata:
            assert metadata["subjects"] == self.subjects

        # add latent code
        self.latent_dim = cfg.get('latent_dim', 0)
        self.identity_dim = cfg.get('identity_dim', 0)
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata["frame_dict"]
            if self.subjects is None:
                self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)
            else:
                self.latent = {
                    subject: nn.Embedding(len(self.frame_dict[subject]), self.latent_dim)
                    for subject in self.subjects
                }
        if self.identity_dim > 0:
            d_cond += self.identity_dim
            self.identity = nn.Embedding(len(metadata['subjects']), self.identity_dim)

        d_out = 3 + 3 + 4
        self.feature_dim = cfg.get('feature_dim', 0)
        d_out += self.feature_dim

        if self.subjects is None:
            self.aabb = metadata['aabb']
        else:
            self.aabb = {}
            for subject in subjects:
                self.aabb[subject] = metadata[subject]['aabb']
        self.hashgrid = HashGrid(cfg.hashgrid)
        self.mlp = VanillaCondMLP(self.hashgrid.n_output_dims, d_cond, d_out, cfg.mlp)

        self.delay = cfg.get('delay', 0)

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        if self.subjects is not None:
            identity_idx = camera.identity_idx
            subject = camera.subject

        if iteration < self.delay:
            deformed_gaussians = gaussians.clone()
            if self.feature_dim > 0:
                setattr(deformed_gaussians, "non_rigid_feature",
                        torch.zeros(gaussians.get_xyz.shape[0], self.feature_dim).cuda())
            return deformed_gaussians, {}

        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs)

        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if self.subjects is None:
                frame_dict = self.frame_dict
            else:
                frame_dict = self.frame_dict[camera.subject]
            if frame_idx not in frame_dict:
                latent_idx = len(frame_dict) - 1
            else:
                latent_idx = frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(pose_feat.device)
            if self.subjects is None:
                latent_code = self.latent(latent_idx)
            else:
                latent_code = self.latent[camera.subject](latent_idx)
            latent_code = latent_code.expand(pose_feat.shape[0], -1)
            pose_feat = torch.cat([pose_feat, latent_code], dim=1)
        if self.identity_dim > 0:
            _identity_idx = torch.Tensor([identity_idx]).long().to(pose_feat.device)
            identity_code = self.identity(_identity_idx)
            #identity_code = gaussians.U2[identity_idx]
            identity_code = identity_code.expand(pose_feat.shape[0], -1)
            pose_feat = torch.cat([pose_feat, identity_code], dim=1)

        xyz = gaussians.get_xyz[:, identity_idx] if self.multigaussian else gaussians.get_xyz
        aabb = self.aabb if self.subjects is None else self.aabb[subject]
        xyz_norm = aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        feature = self.hashgrid(xyz_norm)
        deltas = self.mlp(feature, cond=pose_feat)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]

        if self.multigaussian:
            deformed_gaussians._xyz[:, identity_idx] = gaussians._xyz[:, identity_idx] + delta_xyz
        else:
            deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        scale_offset = self.cfg.get('scale_offset', 'logit')
        if scale_offset == 'logit':
            if self.multigaussian:
                deformed_gaussians._scaling[:, identity_idx] = gaussians._scaling[:, identity_idx] + delta_scale
            else:
                deformed_gaussians._scaling = gaussians._scaling + delta_scale
        elif scale_offset == 'exp':
            if self.multigaussian:
                deformed_gaussians._scaling[:, identity_idx] = torch.log(torch.clamp_min(gaussians.get_scaling[:, identity_idx] + delta_scale, 1e-6))
            else:
                deformed_gaussians._scaling = torch.log(torch.clamp_min(gaussians.get_scaling + delta_scale, 1e-6))            
        elif scale_offset == 'zero':
            delta_scale = torch.zeros_like(delta_scale)
            if self.multigaussian:
                deformed_gaussians._scaling[:, identity_idx] = gaussians._scaling[:, identity_idx]
            else:
                deformed_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        rot_offset = self.cfg.get('rot_offset', 'add')
        if rot_offset == 'add':
            if self.multigaussian:
                deformed_gaussians._rotation[:, identity_idx] = gaussians._rotation[:, identity_idx] + delta_rot
            else:
                deformed_gaussians._rotation = gaussians._rotation + delta_rot
        elif rot_offset == 'mult':
            q1 = delta_rot
            q1[0] = 1.  # [1,0,0,0] represents identity rotation
            delta_rot = delta_rot[1:]
            if self.multigaussian:
                q2 = gaussians._rotation[:, identity_idx]
                deformed_gaussians._rotation[:, identity_idx] = tf.quaternion_multiply(q1, q2)
            else:
                q2 = gaussians._rotation
                # deformed_gaussians._rotation = quaternion_multiply(q1, q2)
                deformed_gaussians._rotation = tf.quaternion_multiply(q1, q2)
        else:
            raise ValueError

        if self.feature_dim > 0:
            setattr(deformed_gaussians, "non_rigid_feature", deltas[:, 10:])

        if compute_loss:
            # regularization
            loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
            loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
            loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            loss_reg = {
                'nr_xyz': loss_xyz,
                'nr_scale': loss_scale,
                'nr_rot': loss_rot
            }
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg

def get_non_rigid_deform(cfg, metadata, subjects=None, multigaussian=False):
    name = cfg.name
    model_dict = {
        "identity": Identity,
        "mlp": MLP,
        "hannw_mlp": HannwMLP,
        "hashgrid": HashGridwithMLP,
    }
    return model_dict[name](cfg, metadata, subjects=subjects, multigaussian=multigaussian)
