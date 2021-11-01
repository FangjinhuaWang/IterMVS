import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)

    def forward(self,x):
        return F.relu(self.conv(x), inplace=True)

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBnReLU(in_planes, planes, 3, stride=stride, pad=1)
        self.conv2 = ConvBn(planes, planes, 3, stride=1, pad=1)

        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:    
            self.downsample = ConvBn(in_planes, planes, 3, stride=stride, pad=1)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x+y)

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size-1, dilation=2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size-1, dilation=2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size-1, dilation=2)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))                
        h = (1-z) * h + z * q

        return h

def differentiable_warping(src_fea, src_proj, ref_proj, depth_samples, return_mask=False):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_samples: [B, Ndepth, H, W] 
    # out: [B, C, Ndepth, H, W]
    batch, num_depth, height, width = depth_samples.size()
    height1, width1 = src_fea.size()[2:]

    with torch.no_grad():
        if batch==2:
            inv_ref_proj = []
            for i in range(batch):
                inv_ref_proj.append(torch.inverse(ref_proj[i]).unsqueeze(0))
            inv_ref_proj = torch.cat(inv_ref_proj, dim=0)
            assert (not torch.isnan(inv_ref_proj).any()), "nan in inverse(ref_proj)"
            proj = torch.matmul(src_proj, inv_ref_proj)
        else:
            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            assert (not torch.isnan(proj).any()), "nan in proj"

        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_samples.device),
                            torch.arange(0, width, dtype=torch.float32, device=depth_samples.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        y = y*(height1/height)
        x = x*(width1/width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(batch, 1, num_depth,
                                                                                            height * width)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # avoid negative depth
        valid_mask = proj_xyz[:, 2:] > 1e-2
        proj_xyz[:, 0:1][~valid_mask] = width
        proj_xyz[:, 1:2][~valid_mask] = height
        proj_xyz[:, 2:3][~valid_mask] = 1
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        valid_mask = valid_mask & (proj_xy[:, 0:1] >=0) & (proj_xy[:, 0:1] < width) \
                    & (proj_xy[:, 1:2] >=0) & (proj_xy[:, 1:2] < height)
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width1 - 1) / 2) - 1 # [B, Ndepth, H*W]
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height1 - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy      

    dim = src_fea.size()[1]
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                padding_mode='zeros',align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, dim, num_depth, height, width)
    if return_mask:
        valid_mask = valid_mask.view(batch,num_depth,height,width)
        return warped_src_fea, valid_mask
    else:
        return warped_src_fea

def upsample(x, upsample_weight, scale=4):
    '''
    upsample the x with convex combination
    '''
    batch, _, height, width = x.size() 

    replicate_pad = nn.ReplicationPad2d(1)
    x = replicate_pad(x)
    x = F.unfold(x, [3,3], padding=0)
    x = x.view(batch,-1,9,1,1,height,width)
    x_upsampled = torch.sum(x*upsample_weight, dim=2)
    x_upsampled = x_upsampled.permute(0,1,4,2,5,3).contiguous().view(batch,-1,scale*height,scale*width)

    return x_upsampled

def depth_normalization(depth, inverse_depth_min, inverse_depth_max):
    '''convert depth map to the index in inverse range'''
    inverse_depth = 1.0 / (depth+1e-5)
    normalized_depth = (inverse_depth - inverse_depth_max) / (inverse_depth_min - inverse_depth_max)
    return normalized_depth

def depth_unnormalization(normalized_depth, inverse_depth_min, inverse_depth_max):
    '''convert the index in inverse range to depth map'''
    inverse_depth = inverse_depth_max + normalized_depth * (inverse_depth_min - inverse_depth_max) # [B,1,H,W]
    depth = 1.0 / inverse_depth
    return depth