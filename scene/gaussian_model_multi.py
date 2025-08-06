import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

import trimesh
import igl

import tensorly as tl
tl.set_backend("pytorch")


class MultiGaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, cfg, subjects):
        self.cfg = cfg

        self.subjects = subjects
        self.N_ids = len(self.subjects)
        self.N_max = 50_000
        self.R = cfg.R
        self.decomp_type = cfg.decomp_type

        # two modes: SH coefficient or feature
        self.use_sh = cfg.use_sh
        self.active_sh_degree = 0
        if self.use_sh:
            self.max_sh_degree = cfg.sh_degree
            self.feature_dim = (self.max_sh_degree + 1) ** 2
        else:
            self.feature_dim = cfg.feature_dim

        self.U1 = torch.empty(0)
        self.U2 = torch.empty(0)
        self.U3 = torch.empty(0)

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def clone(self):
        cloned = MultiGaussianModel(self.cfg, self.subjects)

        properties = ["active_sh_degree",
                      "non_rigid_feature",
                      ]
        for property in properties:
            if hasattr(self, property):
                setattr(cloned, property, getattr(self, property))

        parameters = [
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
            "U1",
            "U2",
            "U3"
        ]
        for parameter in parameters:
            setattr(cloned, parameter, getattr(self, parameter) + 0.)

        return cloned

    def set_fwd_transform(self, T_fwd):
        self.fwd_transform = T_fwd

    # def color_by_opacity(self):
    #     cloned = self.clone()
    #     cloned._features_dc = self.get_opacity.unsqueeze(-1).expand(-1,-1,3)
    #     cloned._features_rest = torch.zeros_like(cloned._features_rest)
    #     return cloned

    def capture(self):
        return (
            self.active_sh_degree,
            self.U1,
            self.U1_xyz,
            self.U1_scaling,
            self.U1_rotation,
            self.U1_feat_dc,
            self.U1_feat_rest,
            self.U1_opacity,
            self.U2,
            self.U3,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self.U1,
            self.U1_xyz,
            self.U1_scaling,
            self.U1_rotation,
            self.U1_feat_dc,
            self.U1_feat_rest,
            self.U1_opacity,
            self.U2,
            self.U3,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        #self.optimizer.load_state_dict(opt_dict)
    
    def get_W(self, identity_idx=None):
        if identity_idx is None:
            _U2 = self.U2
        else:
            _U2 = self.U2[identity_idx:identity_idx+1]

        if self.decomp_type == "Tucker":
            W = tl.tucker_to_tensor((self.weights, [_U2, self.U1, self.U3]))
        elif self.decomp_type == "CP":
            W = torch.matmul(self.U3, tl.tenalg.khatri_rao([_U2, self.U1]).transpose(0, 1))
        elif self.decomp_type == "TensorTrain":
            W = tl.tt_to_tensor((_U2.transpose(0, 1), self.U1.transpose(0, 1), self.U3))

        return W
    
    def update_attributes(self, identity_idx, update_all=True):
        self.U1 = torch.cat(
            (
                self.U1_xyz,
                self.U1_scaling,
                self.U1_rotation,
                self.U1_feat_dc,
                self.U1_feat_rest,
                self.U1_opacity,
            ),
            dim=0
        )
        if update_all:
            # Compute full W and update all
            W = self.get_W().reshape(self.N_max, self.N_ids, -1)
            self._xyz = W[..., :self.dim_1]
            self._scaling = W[..., self.dim_1 : self.dim_2]
            self._rotation = W[..., self.dim_2 : self.dim_3]
            self._features_dc = W[..., self.dim_3 : self.dim_4]
            self._features_rest = W[..., self.dim_4 : self.dim_5]
            self._opacity = W[..., self.dim_5 : self.dim_6]
        else:
            # Use U2 only for identity_idx and update correspondingly
            # Needs retain_graph=True in loss backward in training
            W_i = self.get_W(identity_idx)
            self._xyz[:, identity_idx] = W_i[:, :self.dim_1]
            self._scaling[:, identity_idx] = W_i[:, self.dim_1 : self.dim_2]
            self._rotation[:, identity_idx] = W_i[:, self.dim_2 : self.dim_3]
            self._features_dc[:, identity_idx] = W_i[:, self.dim_3 : self.dim_4]
            self._features_rest[:, identity_idx] = W_i[:, self.dim_4 : self.dim_5]
            self._opacity[:, identity_idx] = W_i[:, self.dim_5 : self.dim_6]
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=-1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, identity_idx, scaling_modifier=1):
        if hasattr(self, 'rotation_precomp'):
            return self.covariance_activation(self.get_scaling[:, identity_idx], scaling_modifier, self.rotation_precomp)
        return self.covariance_activation(self.get_scaling[:, identity_idx], scaling_modifier, self._rotation[:, identity_idx])

    def oneupSHdegree(self):
        if not self.use_sh:
            return
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def get_opacity_loss(self, identity_idx):
        # opacity classification loss
        opacity = self.get_opacity(identity_idx)
        eps = 1e-6
        loss_opacity_cls = -(opacity * torch.log(opacity + eps) + (1 - opacity) * torch.log(1 - opacity + eps)).mean()
        return {'opacity': loss_opacity_cls}

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale=1.):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        if self.use_sh:
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
        else:
            features = torch.zeros((fused_color.shape[0], 1, self.feature_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        # self._rotation = nn.Parameter(rots.requires_grad_(True))
        # self._opacity = nn.Parameter(opacities.requires_grad_(True))

        # Initialize features with rand instead of zeros (can increase R from 10 to 100)
        features = torch.rand((fused_color.shape[0], 1, self.feature_dim)).float().cuda()
        
        init_from_ckpt = True
        W1 = torch.cat((
                fused_point_cloud,
                scales,
                rots,
                features.reshape(features.shape[0], -1),
                opacities,
            ), dim=-1)
        W1 = W1.transpose(0, 1)
        W1 = W1[None] #.repeat(self.N_ids, 1, 1)

        self.dim_1 = fused_point_cloud.shape[-1]
        self.dim_2 = self.dim_1 + scales.shape[-1]
        self.dim_3 = self.dim_2 + rots.shape[-1]
        self.dim_4 = self.dim_3 + features[:, :, 0:1].shape[-1] * features[:, :, 0:1].shape[-2]
        self.dim_5 = self.dim_4 + features[:, :, 1:].shape[-1] * features[:, :, 1:].shape[-2]
        self.dim_6 = self.dim_5 + opacities.shape[-1]

        # Initialization
        # CP gives weights ones (U2), large numbers for both U1 and U3
        # tl.decomposition.parafac(W1, self.R) or tl.decomposition.CP(self.R).fit_transform(W1) same
        # cp_decomp = tl.decomposition.parafac(W1, self.R)  # same with tl.decomposition.CP(self.R).fit_transform(W1)

        # CPPower gives smaller numbers for U1 and U3, U2 sorted - first are large (first *10e3)
        if self.decomp_type == "CP":
            cp_decomp = tl.decomposition.CPPower(self.R).fit_transform(W1)
        elif self.decomp_type == "Tucker":
            cp_decomp = tl.decomposition.Tucker(self.R, n_iter_max=1000, tol=0.00001).fit_transform(W1.repeat(self.N_ids, 1, 1).cpu())  
        elif self.decomp_type == "TensorTrain":
            cp_decomp = tl.decomposition.TensorTrain(self.R).fit_transform(W1.repeat(self.N_ids, 1, 1))
        
        # ConstrainedCP gives small numbers with normalize=True
        # cp_decomp = tl.decomposition.ConstrainedCP(self.R, normalize=True).fit_transform(W1[0].cpu())

        #self.U1, self.U3 = cp_decomp[1]
        if self.decomp_type == "CP" or self.decomp_type == "Tucker":
            self.U2, self.U1, self.U3 = cp_decomp[1]
            self.U1 = self.U1.cuda()
            self.U2 = self.U2.cuda()
            self.U3 = self.U3.cuda()
        elif self.decomp_type == "TensorTrain":
            self.U2, self.U1, self.U3 = cp_decomp[0].cuda(), cp_decomp[1].cuda(), cp_decomp[2].cuda()
            self.U1 = self.U1.transpose(0, 1)
            self.U2 = self.U2.transpose(0, 1)
        print(self.U1.shape, self.U2.shape, self.U3.shape)

        if self.R > 150:
            cp_decomp[0][torch.isnan(cp_decomp[0])] = 1e-20
            self.U1[torch.isnan(self.U1)] = 1e-7
            self.U3[torch.isnan(self.U3)] = 1e-7
            self.U2[torch.isnan(self.U2)] = 1.0

        #self.U2 = torch.ones((self.N_ids, self.R))
        if self.decomp_type == "CP":
            self.U2 = self.U2 * torch.ones((self.N_ids, self.R)).cuda() #* cp_decomp[0]
            self.U3 = self.U3 * cp_decomp[0]
        elif self.decomp_type == "Tucker":    
            self.weights = cp_decomp[0].cuda()  #torch.ones_like(cp_decomp[0].cuda())

        # self.weights = cp_decomp[0].cuda()

        # self.U1 = self.U1.cuda() #nn.Parameter(self.U1.cuda().requires_grad_(True))
        self.U1_xyz = nn.Parameter(self.U1[:self.dim_1].cuda().requires_grad_(True))
        self.U1_scaling = nn.Parameter(self.U1[self.dim_1:self.dim_2].cuda().requires_grad_(True))
        self.U1_rotation = nn.Parameter(self.U1[self.dim_2:self.dim_3].cuda().requires_grad_(True))
        self.U1_feat_dc = nn.Parameter(self.U1[self.dim_3:self.dim_4].cuda().requires_grad_(True))
        self.U1_feat_rest = nn.Parameter(self.U1[self.dim_4:self.dim_5].cuda().requires_grad_(True))
        self.U1_opacity = nn.Parameter(self.U1[self.dim_5:self.dim_6].cuda().requires_grad_(True))
        self.U3 = nn.Parameter(self.U3.cuda().requires_grad_(True))
        
        self.U2 = nn.Parameter(self.U2.cuda().requires_grad_(True))
        #self.U2 = nn.Parameter(cp_decomp[0][None].repeat(self.N_ids, 1).cuda().requires_grad_(True))
        #self.U2 = nn.Parameter(torch.ones((self.N_ids, self.R), device="cuda").requires_grad_(True))  # n_ids x R

        # self.U1 = nn.Parameter(U1[:, :self.R].requires_grad_(True))  # (3 + 7 + 16 * 3 + 1) x R
        # self.U2 = nn.Parameter(torch.rand((self.N_ids, self.R), device="cuda").requires_grad_(True))  # n_ids x R
        # self.U3 = nn.Parameter(torch.rand((self.N_max, self.R), device="cuda").requires_grad_(True))  # n_gaussians x R

        W = self.get_W().reshape(self.N_max, self.N_ids, -1)

        self._xyz = W[..., :self.dim_1]
        self._scaling = W[..., self.dim_1 : self.dim_2]
        self._rotation = W[..., self.dim_2 : self.dim_3]
        self._features_dc = W[..., self.dim_3 : self.dim_4]
        self._features_rest = W[..., self.dim_4 : self.dim_5]
        self._opacity = W[..., self.dim_5 : self.dim_6]

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    @staticmethod
    def kronecker_product(A, B):
        return torch.einsum("ab,cd->acbd", A, B).view(A.size(0) * B.size(0),  A.size(1) * B.size(1))

    @staticmethod
    def khatri_rao_product(matrices):
        n_columns = matrices[0].shape[1]
        n_rows = np.prod([matrix.shape[0] for matrix in matrices])
        kr_product = torch.zeros((n_rows, n_columns)).cuda()
        for i in range(n_columns):
            cum_prod = matrices[0][:, i]  # Accumulates the khatri-rao product of the i-th columns
            for matrix in matrices[1:]:
                cum_prod = torch.einsum('i,j->ij', cum_prod, matrix[:, i]).ravel()
            # the i-th column corresponds to the kronecker product of all the i-th columns of all matrices:
            kr_product[:, i] = cum_prod
        return kr_product

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.U3.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.U3.shape[0], 1), device="cuda")

        feature_ratio = 20.0 if self.use_sh else 1.0
        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self.U1_feat_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self.U1_feat_rest], 'lr': training_args.feature_lr / feature_ratio, "name": "f_rest"},
            {'params': [self.U1_opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.U1_scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self.U1_rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        l += [
     #       {'params': [self.U1_feat_dc], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "f_dc"},
     #       {'params': [self.U1_feat_rest], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "f_rest"},
     #       {'params': [self.U1_opacity], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "opacity"},
     #       {'params': [self.U1_scaling], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "scaling"},
     #       {'params': [self.U1_rotation], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "rotation"},
            {'params': [self.U1_xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "U1"},
            {'params': [self.U2], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "U2"},
            {'params': [self.U3], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "U3"},
         ]
        print([(_l["lr"], _l["name"]) for _l in l])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        finetune_weight = 1.0
        if training_args.finetune:
            # The following lines are used for fine-tuning (personalization) to a specific identity
            finetune_weight = 0.0
            for param_group in self.optimizer.param_groups:
                if not param_group["name"] in ["U2", "f_dc", "f_rest", "opacity"]:
                    param_group['lr'] = 0.0
        self.U1_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale*finetune_weight,
                                                   lr_final=training_args.position_lr_final*self.spatial_lr_scale*finetune_weight,
                                                   lr_delay_mult=training_args.position_lr_delay_mult,
                                                   max_steps=training_args.position_lr_max_steps)
        self.U2_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                   lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                   lr_delay_mult=training_args.position_lr_delay_mult,
                                                   max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step ''' 
        lr = None
        for param_group in self.optimizer.param_groups:
            if iteration % 1000 == 0:
                print(param_group['name'], param_group['lr'])
            if param_group["name"] in ["U1", "U3"]:
                lr = self.U1_scheduler_args(iteration)
                param_group['lr'] = lr
                if iteration % 1000 == 0:
                    print("U1, U3", lr)
            elif param_group["name"] in ["U2"]:
                lr = self.U2_scheduler_args(iteration)
                param_group['lr'] = lr
                if iteration % 1000 == 0:
                    print("U2", lr)
        return lr

    def save_ply(self, path):
        # save_ply for consistency
        # But for the multi-identity model, it saves a .pth
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                "U1": self.U1.detach().cpu().numpy(),
                "U2": self.U2.detach().cpu().numpy(),
                "U3": self.U3.detach().cpu().numpy(),
            },
            path.replace(".ply", ".pth")
        )

    def load_checkpoint(self, path):
        print("Loading {}".format(path))
        (gaussian_params, converter_sd, converter_opt_sd, converter_scd_sd, first_iter) = torch.load(path)
        self.gaussians.restore(gaussian_params, self.cfg.opt)
        self.converter.load_state_dict(converter_sd)
        self.converter.optimizer.load_state_dict(converter_opt_sd)
        self.converter.scheduler.load_state_dict(converter_scd_sd)
        return first_iter

    # def reset_opacity(self):
    #     opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
    #     optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
    #     self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        # load_ply for consistency
        # But for the multi-identity model, it loads a .pth

        ckpt = torch.load(path.replace(".ply", ".pth"))

        U1 = ckpt["U1"]
        U2 = ckpt["U2"]
        U3 = ckpt["U3"]

        # self.U1 = nn.Parameter(torch.tensor(U1, dtype=torch.float, device="cuda").requires_grad_(True))
        self.U1 = U1
        self.U1_xyz = nn.Parameter(self.U1[:self.dim_1].cuda().requires_grad_(True))
        self.U1_scaling = nn.Parameter(self.U1[self.dim_1:self.dim_2].cuda().requires_grad_(True))
        self.U1_rotation = nn.Parameter(self.U1[self.dim_2:self.dim_3].cuda().requires_grad_(True))
        self.U1_feat_dc = nn.Parameter(self.U1[self.dim_3:self.dim_4].cuda().requires_grad_(True))
        self.U1_feat_rest = nn.Parameter(self.U1[self.dim_4:self.dim_5].cuda().requires_grad_(True))
        self.U1_opacity = nn.Parameter(self.U1[self.dim_5:self.dim_6].cuda().requires_grad_(True))
        self.U2 = nn.Parameter(torch.tensor(U2, dtype=torch.float, device="cuda").requires_grad_(True))
        self.U3 = nn.Parameter(torch.tensor(U3, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
