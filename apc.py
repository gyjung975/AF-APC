import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from knn_cuda import KNN
from model.pt2 import Group, fps, knn_point


def random_sample_points_np(points, num):
    """points: [K, 3]"""
    idx = np.random.choice(len(points), num, replace=True)
    return points[idx]


class SORDefense(nn.Module):
    def __init__(self, k=2, alpha=1.1):
        """SOR defense.

        Args:
            k (int, optional): kNN. Defaults to 2.
            alpha (float, optional): \miu + \alpha * std. Defaults to 1.1.
        """
        super(SORDefense, self).__init__()

        self.k = k
        self.alpha = alpha

    def outlier_removal(self, x):
        pc = x.clone().detach().double()
        B, K = pc.shape[:2]
        pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K]
        assert dist.min().item() >= -1e-6
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)  # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # [B, K]
        mean = torch.mean(value, dim=-1)  # [B]
        std = torch.std(value, dim=-1)  # [B]
        threshold = mean + self.alpha * std  # [B]
        bool_mask = (value <= threshold[:, None])  # [B, K]
        sel_pc = [x[i][bool_mask[i]] for i in range(B)]
        return sel_pc

    def forward(self, x):
        with torch.no_grad():
            x = self.outlier_removal(x)
        return x


class APC(nn.Module):
    def __init__(self, k, feat_dim, off_dims, args):
        super().__init__()
        self.k = k
        self.feat_dim = feat_dim
        self.leaky_relu = args.leaky_relu

        self.use_sor = args.sor
        if self.use_sor:
            self.sor = SORDefense(k=args.sor_k, alpha=args.sor_alpha)

        self.prompt_encoder = DGCNNViewLight(self.k, self.feat_dim)

        self.offset = nn.Sequential(
            nn.Conv1d(self.feat_dim, off_dims, 1),
            nn.GELU(),
            nn.Conv1d(off_dims, off_dims, 1),
            nn.GELU(),
            nn.Conv1d(off_dims, 3, 1)
        )

        for l in self.offset:
            if isinstance(l, nn.Conv1d) and l.out_channels == 3:
                nn.init.zeros_(l.weight)
                nn.init.zeros_(l.bias)

    def forward(self, pc):
        B, N, _ = pc.shape
        if pc.shape[-1] == 6:
            pc = pc[:, :, :3]

        if self.use_sor:
            pc = torch.stack(self.sor(pc))

        feat = self.prompt_encoder(pc)
        offset = self.offset(feat)
        offset = offset.transpose(1, 2)

        def_data = pc + offset
        return def_data


class APCT2R(nn.Module):
    def __init__(self, def_num, k, feat_dim, off_dims, args):
        super().__init__()
        self.num_group = args.num_group
        self.group_size = k
        self.def_num = def_num
        self.k = self.group_size
        self.feat_dim = feat_dim

        self.use_sor = args.sor
        if self.use_sor:
            self.sor = SORDefense(k=args.sor_k, alpha=args.sor_alpha)

        self.prompt_encoder = DGCNNViewLightT2(self.k, self.feat_dim)

        self.center_offset = nn.Sequential(
            nn.Conv1d(self.feat_dim * 4, off_dims * 2, 1),
            nn.GELU(),
            nn.Conv1d(off_dims * 2, 3, 1)
            # nn.Conv1d(256, 3 * self.def_num, 1)
        )
        self.nei_offset = nn.Sequential(
            nn.Conv2d(self.feat_dim*4, off_dims*4, 1),
            nn.GELU(),
            nn.Conv2d(off_dims*4, off_dims*2, 1),
            nn.GELU(),
            nn.Conv2d(off_dims*2, 3, 1)
        )

        for m in list(self.center_offset) + list(self.nei_offset):
            if isinstance(m, (nn.Conv1d, nn.Conv2d)) and m.out_channels == 3:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pc):
        B, N, _ = pc.shape
        if pc.shape[-1] == 6:
            pc = pc[:, :, :3]

        if self.use_sor:
            pc = torch.stack(self.sor(pc))

        pc = self._purify(pc)
        return pc

    def _purify(self, pc):
        B, N, _ = pc.shape
        xyz = pc.contiguous()
        center = fps(xyz, self.num_group)
        idx = knn_point(self.group_size, xyz, center)
        idx_flat = idx.view(B, -1)
        neighborhood = torch.gather(xyz, 1, idx_flat.unsqueeze(-1).expand(-1, -1, 3)).view(B, self.num_group, self.group_size, 3)
        neighborhood = neighborhood - center.unsqueeze(2)

        feat = self.prompt_encoder(center, neighborhood)

        offset_cen = self.center_offset(feat.max(-1)[0]).transpose(1, 2)
        offset_nei = self.nei_offset(feat).permute(0, 2, 3, 1).contiguous()
        slot_offset = offset_nei + offset_cen.unsqueeze(2)

        off_sum = torch.zeros(B, N, 3, device=pc.device, dtype=slot_offset.dtype)
        off_sum.scatter_add_(1, idx_flat.unsqueeze(-1).expand(-1, -1, 3), slot_offset.view(B, -1, 3))
        cnt = torch.zeros(B, N, 1, device=pc.device, dtype=slot_offset.dtype)
        cnt.scatter_add_(1, idx_flat.unsqueeze(-1), torch.ones(B, idx_flat.shape[1], 1, device=pc.device, dtype=slot_offset.dtype))
        def_data = pc + off_sum / cnt.clamp(min=1.)
        return def_data


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)       # (bs', #group, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNNViewLight(nn.Module):
    def __init__(self, k, dim):
        super(DGCNNViewLight, self).__init__()
        self.k = k
        self.dim = dim
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)
        self.bn3 = nn.BatchNorm1d(self.dim)

        act_mod = nn.ReLU
        act_mod_args = {}

        self.conv1 = nn.Sequential(nn.Conv2d(6, self.dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args))
        self.conv2 = nn.Sequential(nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   act_mod(**act_mod_args))
        self.fc = nn.Sequential(nn.Conv1d(self.dim*2, self.dim, kernel_size=1, bias=False),
                                self.bn3,
                                act_mod(**act_mod_args))

        self.knn = KNN(k=self.k, transpose_mode=True)

    def forward(self, x):
        batch_size = x.size(0)      # (b, n, 3)
        _, idx = self.knn(x, x)     # (b, n, k)
        x = get_graph_feature(x.transpose(1, 2), k=self.k, idx=idx)     # (b, 3*2, n, k)

        x = self.conv1(x)                       # (b, feat_dim; 256, n, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (b, feat_dim; 256, n)

        xx = self.conv2(x1)                     # (b, feat_dim; 256, n)

        feat = xx.max(dim=-1, keepdim=True)[0]  # (b, feat_dim; 256, 1)
        x2 = torch.cat([x, feat.repeat(1, 1, x.size(-1))], dim=1)   # local + global
        x = self.fc(x2)
        return x


class DGCNNViewLightT2(nn.Module):
    def __init__(self, k, dim):
        super(DGCNNViewLightT2, self).__init__()
        self.k = k
        self.dim = dim
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim*2)
        self.bn3 = nn.BatchNorm1d(self.dim*2)
        self.bn4 = nn.BatchNorm2d(self.dim*4)

        act_mod = nn.ReLU
        act_mod_args = {}

        self.conv1 = nn.Sequential(nn.Conv2d(6, self.dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args))
        self.conv2 = nn.Sequential(nn.Conv2d(self.dim*2, self.dim*2, kernel_size=1, bias=False),
                                   self.bn2,
                                   act_mod(**act_mod_args))
        self.conv3 = nn.Sequential(nn.Conv1d(self.dim*2, self.dim*2, kernel_size=1, bias=False),
                                   self.bn3,
                                   act_mod(**act_mod_args))
        self.fc = nn.Sequential(nn.Conv2d(self.dim*4, self.dim*4, kernel_size=1, bias=False),
                                self.bn4,
                                act_mod(**act_mod_args))

    def forward(self, center, neighborhood):
        batch_size, N, _ = center.size()
        x = center.unsqueeze(2).repeat(1, 1, self.k, 1)
        x = torch.cat((neighborhood, x), dim=3).permute(0, 3, 1, 2).contiguous()

        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=True)[0]

        x2 = torch.cat([x, x1.repeat(1, 1, 1, self.k)], dim=1)
        # x2 = torch.cat([x1.repeat(1, 1, 1, self.k), x], dim=1)

        x = self.conv2(x2)
        x3 = x.max(dim=-1, keepdim=False)[0]

        g = self.conv3(x3)

        feat = g.max(dim=-1, keepdim=True)[0]
        x4 = torch.cat([x, feat.unsqueeze(-1).repeat(1, 1, N, self.k)], dim=1)
        x = self.fc(x4)
        return x
