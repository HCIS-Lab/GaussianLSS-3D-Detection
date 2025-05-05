import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck
from typing import Iterable, Optional
from mmcv.runner import BaseModule

import numpy as np
import math
from einops import rearrange, repeat
import matplotlib.pyplot as plt

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

BottleneckBlock = lambda x: Bottleneck(x, x//4)

class GaussianLSSTransformerEncoder(BaseModule):
    def __init__(self, align_res, in_channels, interm_c, embed_dims, depth_num, depth_start, depth_max, pc_range, init_cfg=None):
        super(GaussianLSSTransformerEncoder, self).__init__(init_cfg)
        self.align_res = AlignRes(**align_res)
        self.depth_num = depth_num
        self.depth_start = depth_start
        self.depth_max = depth_max
        self.error_tolerance = 0.5
        self.pc_range = pc_range

        bins = self.init_bin_centers()
        self.register_buffer('bins', bins, persistent=True)

        in_c = sum(in_channels)
        self.feats = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(interm_c, embed_dims, kernel_size=1, padding=0),
        )
        self.depth = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(interm_c, depth_num, kernel_size=1, padding=0)
        )
        self.opacity = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(interm_c, 1, kernel_size=1, padding=0)
        )
        self.ray_embed = nn.Sequential(
            nn.Linear(3, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(3, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
        )
        self.gs_render = GaussianRenderer(embed_dims, threshold=0.01)

    def init_bin_centers(self):
        """
        depth: b d h w
        """
        depth_range = self.depth_max - self.depth_start
        interval = depth_range / self.depth_num
        interval = interval * torch.ones((self.depth_num+1))
        interval[0] = self.depth_start
        bin_edges = torch.cumsum(interval, 0)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return bin_centers
    
    def pred_depth(self, lidar2img, depth, img_h, img_w, coords_3d=None):
        # b, n, c, h, w = depth.shape
        if coords_3d is None:
            # bins = self.bins * self.bin_scale + self.bin_bias
            coords_3d, coords_d = get_pixel_coords_3d(self.bins, depth, lidar2img, img_h=img_h, img_w=img_w) # b n w h d 3
            coords_3d = rearrange(coords_3d, 'b n w h d c -> (b n) d h w c')
        
        direction_vector = F.normalize((coords_3d[:, 1] - coords_3d[:, 0]), dim=-1)
        depth_prob = depth.softmax(1) # (b n) depth h w
        pred_coords_3d = (depth_prob.unsqueeze(-1) * coords_3d).sum(1)  # (b n) h w 3
        
        delta_3d = pred_coords_3d.unsqueeze(1) - coords_3d
        cov = (depth_prob.unsqueeze(-1).unsqueeze(-1) * (delta_3d.unsqueeze(-1) @ delta_3d.unsqueeze(-2))).sum(1)
        scale = (self.error_tolerance ** 2) / 9 
        cov = cov * scale

        return pred_coords_3d, cov, direction_vector
    
    def get_uncertainty(self, x):
        feats, depth, opacity = self.feats(x), self.depth(x), self.opacity(x).sigmoid()
        return feats, depth, opacity

    def forward(self, mlvl_feats, img_metas):
        mlvl_feats = self.align_res(mlvl_feats)
        x = torch.cat(mlvl_feats, dim=1)
        feats, depth, opacity = self.get_uncertainty(x)

        device = feats.device
        lidar2img = np.asarray([m['lidar2img'] for m in img_metas]).astype(np.float32)
        lidar2img = torch.from_numpy(lidar2img).to(device)  # [B, N, 4, 4]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]
        b, n = lidar2img.shape[:2]

        means3D, cov3D, direction_vector = self.pred_depth(lidar2img, depth, image_h, image_w)
        cov3D = cov3D.flatten(-2, -1)
        cov3D = torch.cat((cov3D[..., 0:3], cov3D[..., 4:6], cov3D[..., 8:9]), dim=-1)

        feats = rearrange(feats, '(b n) d h w -> b (n h w) d', b=b, n=n)
        means3D = rearrange(means3D, '(b n) h w d-> b (n h w) d', b=b, n=n)
        cov3D = rearrange(cov3D, '(b n) h w d -> b (n h w) d',b=b, n=n)
        opacity = rearrange(opacity, '(b n) d h w -> b (n h w) d', b=b, n=n)

        direction_vector = rearrange(direction_vector, '(b n) h w d -> b (n h w) d', b=b, n=n)
        direction_vector = self.ray_embed(direction_vector)

        pos = means3D.clone()
        pos[..., 0] = (pos[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        pos[..., 1] = (pos[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        pos[..., 2] = (pos[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        pos = torch.clamp(pos, 0, 1)
        pos_embed = self.pos_embed(pos)
        
        feats = feats + direction_vector + pos_embed
        x = self.gs_render(feats, means3D, cov3D, opacity)
        return x

class BEVCamera:
    def __init__(self, x_range=(-50, 50), y_range=(-50, 50), image_size=200):
        # Orthographic projection parameters
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.image_width = image_size
        self.image_height = image_size

        # Set up FoV to cover the range [-50, 50] for both X and Y
        self.FoVx = (self.x_max - self.x_min)  # Width of the scene in world coordinates
        self.FoVy = (self.y_max - self.y_min)  # Height of the scene in world coordinates

        # Camera position: placed above the scene, looking down along Z-axis
        self.camera_center = torch.tensor([0, 0, 0], dtype=torch.float32) 

        # Orthographic projection matrix for BEV
        self.set_transform()
    
    def set_transform(self, h=200, w=200, h_meters=100, w_meters=100):
        """ Set up an orthographic projection matrix for BEV. """
        # Create an orthographic projection matrix
        sh = h / h_meters
        sw = w / w_meters
        self.world_view_transform = torch.tensor([
            [ sh,  0.,  0.,         0.],
            [ 0.,  sw,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
        ], dtype=torch.float32)

        self.full_proj_transform = torch.tensor([
            [ sh,  0.,  0.,          h/2.],
            [ 0.,  sw,  0.,          w/2.],
            [ 0.,  0.,  0.,           1.],
            [ 0.,  0.,  0.,           1.],
        ], dtype=torch.float32)

    def set_size(self, h, w):
        self.image_height = h
        self.image_width = w

class GaussianRenderer(nn.Module):
    def __init__(self, embed_dims, threshold=0.05):
        super().__init__()
        self.viewpoint_camera = BEVCamera()
        self.rasterizer = GaussianRasterizer()
        self.embed_dims = embed_dims
        self.threshold = threshold
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, features, means3D, cov3D, opacities):
        """
        features: b G d
        means3D: b G 3
        uncertainty: b G 6
        opacities: b G 1
        """ 
        b = features.shape[0]
        device = means3D.device

        bev_out = []
        mask = (opacities > self.threshold)
        mask = mask.squeeze(-1)
        self.set_render_scale(200, 200)
        self.set_Rasterizer(device)

        for i in range(b):
            rendered_bev, _ = self.rasterizer(
                means3D=means3D[i][mask[i]],
                means2D=None,
                shs=None,  # No SHs used
                colors_precomp=features[i][mask[i]],
                opacities=opacities[i][mask[i]],
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D[i][mask[i]]
            )
            bev_out.append(rendered_bev)
            
        x = torch.stack(bev_out, dim=0) # b d h w
        x = self.conv(x)
        return x
        
    @torch.no_grad()
    def set_Rasterizer(self, device):
        tanfovx = math.tan(self.viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(self.viewpoint_camera.FoVy * 0.5)

        bg_color = torch.zeros((self.embed_dims)).to(device) 
        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.viewpoint_camera.image_height),
            image_width=int(self.viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1,
            viewmatrix=self.viewpoint_camera.world_view_transform.to(device),
            projmatrix=self.viewpoint_camera.full_proj_transform.to(device),
            sh_degree=0,  # No SHs used 
            campos=self.viewpoint_camera.camera_center.to(device),
            prefiltered=False,
            debug=False
        )
        self.rasterizer.set_raster_settings(raster_settings)

    @torch.no_grad()
    def set_render_scale(self, h, w):
        self.viewpoint_camera.set_size(h, w)
        self.viewpoint_camera.set_transform(h, w)

class AlignRes(BaseModule):
    """Align resolutions of the outputs of the backbone."""

    def __init__(
        self,
        mode="upsample",
        scale_factors: Iterable[int] = [1, 2],
        in_channels: Iterable[int] = [256, 512, 1024, 2048],
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        if mode == "upsample":
            for s in scale_factors:
                if s != 1:
                    self.layers.append(
                        nn.Upsample(
                            scale_factor=s, mode="bilinear", align_corners=False
                        )
                    )
                else:
                    self.layers.append(nn.Identity())

        elif mode == "conv2dtranspose":
            for i, in_c in enumerate(in_channels):
                if scale_factors[i] != 1:
                    self.layers.append(
                        nn.ConvTranspose2d(
                            in_c, in_c, kernel_size=2, stride=2, padding=0
                        )
                    )
                else:
                    self.layers.append(nn.Identity())

        else:
            raise NotImplementedError
        return

    def forward(self, x):
        x = [_.flatten(0,1) for _ in x]
        return [self.layers[i](xi) for i, xi in enumerate(x)]

@torch.no_grad()
def get_pixel_coords_3d(coords_d, depth, lidar2img, img_h=224, img_w=480):
    eps = 1e-5
    
    B, N = lidar2img.shape[:2]
    H, W = depth.shape[-2:]
    # scale = img_h // H
    coords_h = torch.linspace(0, 1, H, device=depth.device).float() * img_h
    coords_w = torch.linspace(0, 1, W, device=depth.device).float() * img_w

    D = coords_d.shape[0]
    coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
    coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
    coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
    img2lidars = lidar2img.inverse() # b n 4 4

    coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
    img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
    coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] # B N W H D 3

    return coords3d, coords_d