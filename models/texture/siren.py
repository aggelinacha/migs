# SIREN MLP with latent modulations
# From data to functa

# PyTorch generated from ChatGPT
# Original code in JAX: https://github.com/google-deepmind/functa/blob/main/function_reps.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, Optional, Tuple


class Sine(nn.Module):
    """Applies a scaled sine transform to input: out = sin(w0 * in)."""

    def __init__(self, w0: float = 1.):
        """Constructor.

        Args:
            w0 (float): Scale factor in sine activation (omega_0 factor from SIREN).
        """
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class FiLM(nn.Module):
    """Applies a FiLM modulation: out = scale * in + shift.

    Notes:
        We currently initialize FiLM layers as the identity. However, this may not
        be optimal. In pi-GAN for example they initialize the layer with a random
        normal.
    """
    def __init__(self, f_in: int, modulate_scale: bool = True, modulate_shift: bool = True):
        """Constructor.

        Args:
            f_in: Number of input features.
            modulate_scale: If True, modulates scales.
            modulate_shift: If True, modulates shifts.
        """
        super(FiLM, self).__init__()
        assert modulate_scale or modulate_shift
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.scale = nn.Parameter(torch.ones(f_in)) if modulate_scale else 1.
        self.shift = nn.Parameter(torch.zeros(f_in)) if modulate_shift else 0.

    def forward(self, x):
        return self.scale * x + self.shift


class ModulatedSirenLayer(nn.Module):
    """Applies a linear layer followed by a modulation and sine activation."""

    def __init__(self, f_in: int, f_out: int, w0: float = 1., is_first: bool = False, is_last: bool = False,
                 modulate_scale: bool = True, modulate_shift: bool = True, apply_activation: bool = True):
        """Constructor.

        Args:
            f_in (int): Number of input features.
            f_out (int): Number of output features.
            w0 (float): Scale factor in sine activation.
            is_first (bool): Whether this is first layer of model.
            is_last (bool): Whether this is last layer of model.
            modulate_scale: If True, modulates scales.
            modulate_shift: If True, modulates shifts.
            apply_activation: If True, applies sine activation.
        """
        super(ModulatedSirenLayer, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.apply_activation = apply_activation
        self.init_range = 1 / f_in if is_first else (6 / f_in) ** 0.5 / w0

        self.linear = nn.Linear(f_in, f_out)
        nn.init.uniform_(self.linear.weight, -self.init_range, self.init_range)
        nn.init.uniform_(self.linear.bias, -self.init_range, self.init_range)

        if self.is_last:
            self.shift = 0.5
        else:
            if self.modulate_scale or self.modulate_shift:
                self.film = FiLM(f_out, modulate_scale, modulate_shift)

            if self.apply_activation:
                self.sine = Sine(w0)

    def forward(self, x):
        x = self.linear(x)
        if self.is_last:
            return x + self.shift
        else:
            if self.modulate_scale or self.modulate_shift:
                x = self.film(x)
            if self.apply_activation:
                x = self.sine(x)
            return x


class MetaSGDLrs(nn.Module):
    """Module storing learning rates for meta-SGD.

    Notes:
        This module does not apply any transformation but simply stores the learning
        rates. Since we also learn the learning rates we treat them the same as
        model params.
    """

    def __init__(self, num_lrs: int, lrs_init_range: Tuple[float, float] = (0.005, 0.1), lrs_clip_range: Tuple[float, float] = (-5., 5.)):
        """Constructor.

        Args:
            num_lrs: Number of learning rates to learn.
            lrs_init_range: Range from which initial learning rates will be uniformly sampled.
            lrs_clip_range: Range at which to clip learning rates. Default value will effectively avoid any clipping, but typically learning rates should be positive and small.
        """
        super(MetaSGDLrs, self).__init__()
        self.num_lrs = num_lrs
        self.lrs_clip_range = lrs_clip_range

        self.meta_sgd_lrs = nn.Parameter(torch.empty(num_lrs))
        nn.init.uniform_(self.meta_sgd_lrs, *lrs_init_range)

    def forward(self):
        return torch.clamp(self.meta_sgd_lrs, *self.lrs_clip_range)
    

class ModulatedSiren(nn.Module):
    """SIREN model with FiLM modulations as in pi-GAN."""

    def __init__(self, width: int = 256, depth: int = 5, out_channels: int = 3, w0: float = 1., modulate_scale: bool = True,
                 modulate_shift: bool = True, use_meta_sgd: bool = False, meta_sgd_init_range: Tuple[float, float] = (0.005, 0.1),
                 meta_sgd_clip_range: Tuple[float, float] = (-5., 5.)):
        """Constructor.

        Args:
            width (int): Width of each hidden layer in MLP.
            depth (int): Number of layers in MLP.
            out_channels (int): Number of output channels.
            w0 (float): Scale factor in sine activation in first layer.
            modulate_scale: If True, modulates scales.
            modulate_shift: If True, modulates shifts.
            use_meta_sgd: Whether to use meta-SGD.
            meta_sgd_init_range: Range from which initial meta_sgd learning rates will be uniformly sampled.
            meta_sgd_clip_range: Range at which to clip learning rates.
        """
        super(ModulatedSiren, self).__init__()
        self.width = width
        self.depth = depth
        self.out_channels = out_channels
        self.w0 = w0
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.use_meta_sgd = use_meta_sgd
        self.meta_sgd_init_range = meta_sgd_init_range
        self.meta_sgd_clip_range = meta_sgd_clip_range

        if self.use_meta_sgd:
            self.modulations_per_unit = int(modulate_scale) + int(modulate_shift)
            self.num_modulations = width * (depth - 1) * self.modulations_per_unit
            self.meta_sgd_lrs = MetaSGDLrs(self.num_modulations, self.meta_sgd_init_range, self.meta_sgd_clip_range)

        self.layers = nn.ModuleList()
        self.layers.append(ModulatedSirenLayer(2, self.width, is_first=True, w0=self.w0, modulate_scale=self.modulate_scale, modulate_shift=self.modulate_shift))

        for _ in range(1, self.depth - 1):
            self.layers.append(ModulatedSirenLayer(self.width, self.width, w0=self.w0, modulate_scale=self.modulate_scale, modulate_shift=self.modulate_shift))

        self.layers.append(ModulatedSirenLayer(self.width, self.out_channels, is_last=True, w0=self.w0, modulate_scale=self.modulate_scale, modulate_shift=self.modulate_shift))

    def forward(self, coords):
        x = coords.view(-1, coords.shape[-1])

        for layer in self.layers:
            x = layer(x)

        return x.view(*coords.shape[:-1], self.out_channels)


class LatentVector(nn.Module):
    """Module that holds a latent vector.
    
    Notes:
        This module does not apply any transformation but simply stores a latent
        vector. This is to make sure that all data necessary to represent an image
        (or a NeRF scene or a video) is present in the model params. This also makes
        it easier to use the partition_params function.
    """

    def __init__(self, latent_dim: int, latent_init_scale: float = 0.0):
        """Constructor.

        Args:
            latent_dim: Dimension of latent vector.
            latent_init_scale: Scale at which to randomly initialize latent vector.
        """
        super(LatentVector, self).__init__()
        self.latent_vector = nn.Parameter(torch.empty(latent_dim))
        nn.init.uniform_(self.latent_vector, -latent_init_scale, latent_init_scale)

    def forward(self):
        return self.latent_vector


class LatentToModulation(nn.Module):
    """Function mapping latent vector to a set of modulations."""

    def __init__(self,
                 latent_dim: int,
                 layer_sizes: Tuple[int, ...],
                 width: int,
                 num_modulation_layers: int,
                 modulate_scale: bool = True,
                 modulate_shift: bool = True,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        """Constructor.

        Args:
            latent_dim: Dimension of latent vector.
            layer_sizes: List of hidden layer sizes for MLP parameterizing the map
                from latent to modulations. Input dimension is inferred from latent_dim
                and output dimension is inferred from number of modulations.
            width: Width of each hidden layer in MLP of function rep.
            num_modulation_layers: Number of layers in MLP that contain modulations.
            modulate_scale: If True, returns scale modulations.
            modulate_shift: If True, returns shift modulations.
            activation: Activation function to use in MLP.
        """
        super(LatentToModulation, self).__init__()
        # Must modulate at least one of shift and scale
        assert modulate_scale or modulate_shift

        self.latent_dim = latent_dim
        self.layer_sizes = layer_sizes
        self.width = width
        self.num_modulation_layers = num_modulation_layers
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift

        # MLP outputs all modulations. We apply modulations on every hidden unit
        # (i.e on width number of units) at every modulation layer.
        # At each of these we apply either a scale or a shift or both,
        # hence total output size is given by following formula
        self.modulations_per_unit = int(modulate_scale) + int(modulate_shift)
        self.modulations_per_layer = width * self.modulations_per_unit
        self.output_size = num_modulation_layers * self.modulations_per_layer

        layers = []
        input_size = latent_dim
        for size in layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(activation)
            input_size = size

        layers.append(nn.Linear(input_size, self.output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, latent_vector: torch.Tensor) -> Dict[int, Dict[str, torch.Tensor]]:
        modulations = self.network(latent_vector)
        # Partition modulations into scales and shifts at every layer
        outputs = {}
        for i in range(self.num_modulation_layers):
            single_layer_modulations = {}
            start = i * self.width * self.modulations_per_unit
            # Note that we add 1 to scales so that outputs of MLP will be centered
            # (since scale = 1 corresponds to identity function)
            if self.modulate_scale:
                single_layer_modulations['scale'] = modulations[..., start:start + self.width] + 1
                start += self.width
            if self.modulate_shift:
                single_layer_modulations['shift'] = modulations[..., start:start + self.width]
            outputs[i] = single_layer_modulations
        return outputs


class LatentModulatedSiren(nn.Module):
    """SIREN model with FiLM modulations generated from a latent vector."""

    def __init__(self,
                 width: int = 256,
                 depth: int = 5,
                 in_channels: int = 128,
                 out_channels: int = 3,
                 latent_dim: int = 64,
                 n_latents: int = 1,
                 layer_sizes: Tuple[int, ...] = (256, 512),
                 w0: float = 1.,
                 modulate_scale: bool = True,
                 modulate_shift: bool = True,
                 latent_init_scale: float = 0.01,
                 use_meta_sgd: bool = False,
                 meta_sgd_init_range: Tuple[float, float] = (0.005, 0.1),
                 meta_sgd_clip_range: Tuple[float, float] = (-5., 5.)):
        """Constructor.

        Args:
            width (int): Width of each hidden layer in MLP.
            depth (int): Number of layers in MLP.
            out_channels (int): Number of output channels.
            latent_dim: Dimension of latent vector.
            layer_sizes: List of hidden layer sizes for MLP parameterizing the map
                from latent to modulations. Input dimension is inferred from latent_dim
                and output dimension is inferred from number of modulations.
            w0 (float): Scale factor in sine activation in first layer.
            modulate_scale: If True, modulates scales.
            modulate_shift: If True, modulates shifts.
            latent_init_scale: Scale at which to randomly initialize latent vector.
            use_meta_sgd: Whether to use meta-SGD.
            meta_sgd_init_range: Range from which initial meta_sgd learning rates will
                be uniformly sampled.
            meta_sgd_clip_range: Range at which to clip learning rates.
        """
        super(LatentModulatedSiren, self).__init__()
        self.width = width
        self.depth = depth
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.layer_sizes = layer_sizes
        self.w0 = w0
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.latent_init_scale = latent_init_scale
        self.use_meta_sgd = use_meta_sgd
        self.meta_sgd_init_range = meta_sgd_init_range
        self.meta_sgd_clip_range = meta_sgd_clip_range

        # Initialize meta-SGD learning rates
        if self.use_meta_sgd:
            self.meta_sgd_lrs = MetaSGDLrs(self.latent_dim,
                                           self.meta_sgd_init_range,
                                           self.meta_sgd_clip_range)

        # Initialize latent vector and map from latents to modulations
        if n_latents == 1:
            # original
            self.latent = LatentVector(latent_dim, latent_init_scale)
        else:
            self.latent = nn.Embedding(n_latents, latent_dim)

        self.latent_to_modulation = LatentToModulation(
            latent_dim=latent_dim,
            layer_sizes=layer_sizes,
            width=width,
            num_modulation_layers=depth - 1,
            modulate_scale=modulate_scale,
            modulate_shift=modulate_shift
        )

        self.siren_layers = nn.ModuleList()
        # Initial layer (note all modulations are set to False here, since we
        # directly apply modulations from latent_to_modulations output).
        self.siren_layers.append(
            ModulatedSirenLayer(
                f_in=in_channels,
                f_out=self.width,
                is_first=True,
                w0=self.w0,
                modulate_scale=False,
                modulate_shift=False,
                apply_activation=False
            )
        )
        # Hidden layers
        for _ in range(1, self.depth - 1):
            self.siren_layers.append(
                ModulatedSirenLayer(
                    f_in=self.width,
                    f_out=self.width,
                    w0=self.w0,
                    modulate_scale=False,
                    modulate_shift=False,
                    apply_activation=False
                )
            )
        # Final layer
        self.siren_layers.append(
            ModulatedSirenLayer(
                f_in=self.width,
                f_out=self.out_channels,
                is_last=True,
                w0=self.w0,
                modulate_scale=False,
                modulate_shift=False
            )
        )

    def modulate(self, x: torch.Tensor, modulations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Modulates input according to modulations.

        Args:
            x: Hidden features of MLP.
            modulations: Dict with keys 'scale' and 'shift' (or only one of them)
                containing modulations.

        Returns:
            Modulated vector.
        """
        if 'scale' in modulations:
            x = modulations['scale'] * x
        if 'shift' in modulations:
            x = x + modulations['shift']
        return x

    def forward(self, coords: torch.Tensor, latent_idx=None) -> torch.Tensor:
        """Evaluates model at a batch of coordinates.

        Args:
            coords (torch.Tensor): Tensor of coordinates. Should have shape (height, width, 2)
                for images and (depth/time, height, width, 3) for 3D shapes/videos.

        Returns:
            Output features at coords.
        """
        # Compute modulations based on latent vector
        if latent_idx is None:
            # original
            latent_vector = self.latent()
        else:
            latent_vector = self.latent(latent_idx)

        modulations = self.latent_to_modulation(latent_vector)

        # Flatten coordinates
        x = coords.view(-1, coords.shape[-1])

        # Initial layer
        x = self.siren_layers[0](x)
        x = self.modulate(x, modulations[0])
        x = Sine(self.w0)(x)

        # Hidden layers
        for i in range(1, self.depth - 1):
            x = self.siren_layers[i](x)
            x = self.modulate(x, modulations[i])
            x = Sine(self.w0)(x)

        # Final layer
        out = self.siren_layers[-1](x)

        # Unflatten output
        return out.view(*coords.shape[:-1], self.out_channels)


def get_num_weights_and_modulations(params):
    """Returns the number of weights and modulations of ModulatedSiren model."""
    weights, modulations = partition_params(params)
    num_weights = sum(p.numel() for p in weights if p.requires_grad)
    num_modulations = sum(p.numel() for p in modulations if p.requires_grad)
    return num_weights, num_modulations


def partition_params(params):
    """Partitions ModulatedSiren parameters into weights and modulations."""
    weights = []
    modulations = []
    for name, param in params.items():
        if 'fi_lm' in name or 'latent_vector' in name:
            modulations.append(param)
        else:
            weights.append(param)
    return weights, modulations


def partition_shared_params(shared_params):
    """Partitions shared parameters into weights and learning rates."""
    weights = []
    lrs = []
    for name, param in shared_params.items():
        if 'meta_sgd_lrs' in name:
            lrs.append(param)
        else:
            weights.append(param)
    return weights, lrs


def merge_params(weights, modulations):
    """Merges weights and modulations into a single set of parameters."""
    return {**dict(modulations), **dict(weights)}


def update_params(params, modulation):
    """Update ModulatedSiren parameters by only updating modulations."""
    weights, init_modulation = partition_params(params)
    modulation_tree = torch.cat([mod.view(-1) for mod in init_modulation])
    modulated_params = merge_params(weights, modulation_tree)
    return modulated_params


def get_coordinate_grid(res, centered=True):
    """Returns a normalized coordinate grid for a res by res sized image."""
    if centered:
        half_pixel = 1. / (2. * res)
        coords_one_dim = torch.linspace(half_pixel, 1. - half_pixel, res)
    else:
        coords_one_dim = torch.linspace(0, 1, res)
    return torch.stack(torch.meshgrid(coords_one_dim, coords_one_dim), dim=-1)
