import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
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


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBnReLU(in_planes, planes, 3, stride=stride, pad=1)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_planes,planes,3,stride,1),
        #     nn.InstanceNorm2d(planes),
        #     nn.ReLU(inplace=True)
        # )
        self.conv2 = ConvBn(planes, planes, 3, stride=1, pad=1)
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(planes,planes,3,1,1),
        #     nn.InstanceNorm2d(planes),
        # )
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:    
            self.downsample = ConvBn(in_planes, planes, 3, stride=stride, pad=1)
            # self.downsample = nn.Sequential(
            #     nn.Conv2d(in_planes,planes,3,stride,1),
            #     nn.InstanceNorm2d(planes),
            # )

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
        # self.w = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0)

        # self.convz_glo = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0)
        # self.convr_glo = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0)
        # self.convq_glo = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0)

    def forward(self, h, x, weight=None):
        batch, hidden_dim = h.size()[0:2]
        hx = torch.cat([h, x], dim=1)

        # global feature vector
        # glo = torch.sigmoid(self.w(h)) * h
        # glo = glo.view(batch, hidden_dim, -1).mean(-1).view(batch, hidden_dim, 1, 1)

        # z = torch.sigmoid(self.convz(hx) + self.convz_glo(glo))
        # r = torch.sigmoid(self.convr(hx) + self.convr_glo(glo))
        # q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + self.convq_glo(glo))
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))                
        h = (1-z) * h + z * q
        
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(SepConvGRU, self).__init__()

        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,3), padding=(0,1), dilation=(1,1))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,3), padding=(0,1), dilation=(1,1))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,3), padding=(0,1), dilation=(1,1))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (3,1), padding=(1,0), dilation=(1,1))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (3,1), padding=(1,0), dilation=(1,1))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (3,1), padding=(1,0), dilation=(1,1))


    def forward(self, h, x, weight):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        hx = hx * weight
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        hx = hx * weight
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h


class ConvLSTM(nn.Module):

    def __init__(self, hidden_dim, input_dim, kernel_size=3, bias=False):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTM, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              dilation=1,
                              bias=bias)

    def forward(self, cur_state, input, weight=None):
        
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined*weight)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


def point_normalization(points):
    # points: [B,N,3]
    batch, num, _ = points.size()
    # print(points[0,:,:])
    centroid = torch.mean(points, dim=1, keepdim=True) #[B,1,3]
    # print(centroid[0,:,:])
    points = points - centroid
    dists = (torch.sum(points ** 2, dim=2)).sqrt() # [B,N]
    mean_dist = torch.mean(dists, dim=1, keepdim=True) #[B,1]
    scale = 3**0.5 / mean_dist
    points=points*scale.unsqueeze(2)
    # print(points[0,:,:])
    return points

def positionalencoding1d(x, L=2):
    batch, dim, height, width = x.size()
    # print(x.size())
    # [B,2CL]
    pe = torch.zeros((batch, dim*2*L, height, width), device=x.device)
    # print(pe.size())
    coeff = 2**(torch.arange(0,L, device=x.device).float()) * math.pi
    # print(coeff.get_device())
    # [B,L]
    # coeff = coeff.unsqueeze(0).repeat(batch,1, height, width)
    coeff = coeff.view(1,L,1,1)
    for i in range(dim):
        pe[:,i::2*dim] = torch.sin(x[:,i:i+1] * coeff)
        pe[:,dim+i::2*dim] = torch.cos(x[:,i:i+1] * coeff)
    # [B,2CL,H,W]
    return pe

def positionalencoding2d(dim, height, width, device):
    
    pe = torch.zeros(dim, height, width).to(device)
    L = dim//4
    # each dimension has dim//2 encodings
    dim = dim//2
    
    pos_w = torch.arange(0, width).float().unsqueeze(1)
    pos_w = pos_w * 2 / ((width - 1)) - 1
    pos_h = torch.arange(0, height).float().unsqueeze(1)
    pos_h = pos_h * 2 / ((height - 1)) - 1
    coeff = 2**(torch.arange(0,L).float()) * math.pi
    pe[0:dim:2, :, :] = torch.sin(pos_w * coeff).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:dim:2, :, :] = torch.cos(pos_w * coeff).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[dim::2, :, :] = torch.sin(pos_h * coeff).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[dim+1::2, :, :] = torch.cos(pos_h * coeff).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    # [1,C,H,W]
    return pe.unsqueeze(0)


def attention(query, key, value):
    dim = query.size()[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5 #[B,N,N_query,N_key]
    prob = torch.nn.functional.softmax(scores, dim=-1) #[B,N,N_query,N_key]
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob #[B,C//N,N,N_query]

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads, feature_dim):
        super().__init__()
        assert feature_dim % num_heads == 0
        self.dim = feature_dim // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.proj = nn.ModuleList([nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
                                    nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
                                    nn.Conv1d(feature_dim, feature_dim, kernel_size=1)])

    def forward(self, query, key, value):
        batch = query.size()[0]
        
        query, key, value = [l(x).view(batch, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))] #[B,C//N,N,N_query(H*W)]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch, self.dim*self.num_heads, -1))

class AttenPropagation(nn.Module):
    def __init__(self, feature_dim, num_heads, height, width):
        super(AttenPropagation, self).__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP(feature_dim, feature_dim, [2*feature_dim])
        # nn.init.constant_(self.mlp[-1].bias, 0.0)
        self.norm1 = torch.nn.LayerNorm([height*width], elementwise_affine=False)
        self.norm2 = torch.nn.LayerNorm([height*width], elementwise_affine=False)
        # self.norm1 = nn.BatchNorm1d(feature_dim)
        # self.norm2 = nn.BatchNorm1d(feature_dim)

    def forward(self, fea_pixel, fea_keypoint):
        attn = self.attn(fea_pixel, fea_keypoint, fea_keypoint)
        fea_pixel = self.norm1(fea_pixel + attn)
        mlp = self.mlp(fea_pixel.permute(0,2,1))
        return self.norm2(fea_pixel + mlp.permute(0,2,1))
        # fea_pixel = fea_pixel + attn
        # mlp = self.mlp(fea_pixel.permute(0,2,1))
        # return fea_pixel + mlp.permute(0,2,1)

def ViewAngleWeight(relative_extrinsic, intrinsics_inv, depth_sample):
    # relative_extrinsic: [B,4,4]
    # intrinsics: [B,3,3]
    # depth_sample: [B.Ndepth,H,W]
    theta = 5
    sigma = 5
    batch, num_depth, height, width = depth_sample.size()
    src_camera = relative_extrinsic[:,0:3,3] #[B,3]
    with torch.no_grad():
        y, x = torch.meshgrid(
                    [torch.arange(0, height, dtype=torch.float32, device=depth_sample.device),
                    torch.arange(0, width, dtype=torch.float32, device=depth_sample.device)]
                )
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        xyz = torch.matmul(intrinsics_inv, xyz) 
        # position of each pixel in reference camera frame
        P = xyz.unsqueeze(2) * depth_sample.view(batch, 1, num_depth, height * width) # [B, 3, Ndepth, H*W]
        # assert (not torch.isnan(P).any()), "nan in P"
        P_normalize = F.normalize(P, dim=1)
        # assert (not torch.isnan(P_normalize).any()), "nan in P_normalize"
        src2P = P - src_camera.view(batch,3,1,1)
        # assert (not torch.isnan(src2P).any()), "nan in src2P"
        src2P_normalize = F.normalize(src2P, dim=1)
        # assert (not torch.isnan(src2P_normalize).any()), "nan in src2P_normalize"
        dot_product = torch.sum(P_normalize * src2P_normalize,dim=1)
        dot_product = torch.clamp(dot_product, min=-1+1e-6, max=1-1e-6)
        view_angle = 180 / math.pi * torch.acos(dot_product)
        # assert (not torch.isnan(view_angle).any()), "nan in view_angle"
        view_angle = view_angle.view(batch,num_depth,height,width)
        view_angle_weight = torch.exp(-(view_angle-theta)**2 / (2* (sigma**2)))
        # print(view_angle[0,:,height//2,width//2])
    return view_angle_weight


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
        
        # assert (not torch.isnan(proj).any()), "nan in proj"
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        # if xyz==None:
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

def differentiable_warping_grid(src_proj, ref_proj, depth_samples):
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_samples: [B, Ndepth, H, W]
    batch, num_depth, height, width = depth_samples.size()

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
        
        # assert (not torch.isnan(proj).any()), "nan in proj"
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        # if xyz==None:
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_samples.device),
                            torch.arange(0, width, dtype=torch.float32, device=depth_samples.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        
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
    proj_xy = proj_xy.view(batch,2,height,width)
    valid_mask = valid_mask.view(batch,1,height,width)
    # assert (not torch.isnan(proj_xy).any()), "nan in proj_xy"

    return proj_xy, valid_mask
    

def normal_estimation(depth, instrinsic_inv, kernel=5, confidence=None):
    '''
    compute normal from depth
    '''
    batch, _, height, width = depth.size()
    device = depth.get_device()
    
    replicate_pad = nn.ReplicationPad2d(kernel//2)
    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                            torch.arange(0, width, dtype=torch.float32, device=device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        xyz = torch.matmul(instrinsic_inv, xyz) # [B, 3, H*W]
    
    points = xyz * depth.view(batch, -1, height*width)  # [B, 3, H*W]
    points = points.view(batch, 3, height, width)    # [B,3,H,W]
    points_pad = replicate_pad(points)
    
    point_matrix = F.unfold(points_pad, kernel_size=kernel, stride=1, padding=0, dilation=1)
    '''
    confidence_pad = replicate_pad(confidence)
    W = F.unfold(confidence_pad, kernel_size=kernel, stride=1, padding=0, dilation=1)
    W = W+1e-5
    '''

    # An = b
    A = point_matrix.view(batch, 3, kernel**2, height, width)  # [B, 3, 9, H, W]
    A = A.permute(0, 3, 4, 2, 1) # [B, H, W, 9, 3]
    '''
    W = W.view(batch, kernel**2, height, width)  # [B, 9, H, W]
    W = W.permute(0,2,3,1)
    W = torch.diag_embed(W)
    '''
    # print(W[0,0,0,0:5,0:5])
    # print(W.size())
    A_trans = A.transpose(3, 4)  # [B, H, W, 3, 9]
    b = torch.ones([batch, height, width, kernel**2, 1], device=device) #[B, H, W, 9, 1]

    # A.T W A
    '''
    point_multi = torch.matmul(torch.matmul(A_trans, W), A) #[B, H, W, 3, 3]
    '''
    point_multi = torch.matmul(A_trans, A)
    
    # diag = torch.eye(3, device=device).view(1,1,1,3,3) #[1,1,1,3,3]
    # diag = diag.repeat(batch,1,1,1,1)           #[B,1,1,3,3]
    # inversible matrix
    try:
        inversible_matrix = point_multi+1e-3
        if batch==2:
            inv_matrix = []
            for i in range(batch):
                inv_matrix.append(torch.inverse(inversible_matrix[i]).unsqueeze(0))
            inv_matrix = torch.cat(inv_matrix, dim=0)
            assert (not torch.isnan(inv_matrix).any()), "nan in inv_matrix"
        else:
            inv_matrix = torch.inverse(inversible_matrix.to(device))
            assert (not torch.isnan(inv_matrix).any()), "nan in inv_matrix"
    except:
        inversible_matrix = point_multi+1e-1
        if batch==2:
            inv_matrix = []
            for i in range(batch):
                inv_matrix.append(torch.inverse(inversible_matrix[i]).unsqueeze(0))
            inv_matrix = torch.cat(inv_matrix, dim=0)
            assert (not torch.isnan(inv_matrix).any()), "nan in inv_matrix"
        else:
            inv_matrix = torch.inverse(inversible_matrix.to(device))
            assert (not torch.isnan(inv_matrix).any()), "nan in inv_matrix"

    # n = (A.T W A)^-1 A.T W b // || (A.T W A)^-1 A.T W b ||2
    '''
    W_b = torch.matmul(W, b)
    generated_norm = torch.matmul(torch.matmul(inv_matrix, A_trans), W_b).squeeze(4) #[B,H,W,3]
    '''
    generated_norm = torch.matmul(torch.matmul(inv_matrix, A_trans), b).squeeze(4) #[B,H,W,3]
    normal = F.normalize(generated_norm, p=2, dim=3)
    # [B,3,H,W]
    normal = normal.permute(0,3,1,2).contiguous()
    return normal



def normal_update_depth(depth, instrinsic_inv, depth_min, depth_max, kernel=3, kernel_1=5, device="cuda"):
    '''
    compute normal from depth
    then update depth with normal
    '''
    batch, _, height, width = depth.size()
    

    with torch.no_grad():
        replicate_pad = nn.ReplicationPad2d(kernel//2)

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                            torch.arange(0, width, dtype=torch.float32, device=device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        xyz = torch.matmul(instrinsic_inv, xyz) # [B, 3, H*W]
        points = xyz * depth.view(batch, -1, height*width)  # [B, 3, H*W]
        points = points.view(batch, 3, height, width)    # [B,3,H,W]
        points_pad = replicate_pad(points)
        point_matrix = F.unfold(points_pad, kernel_size=kernel, stride=1, padding=0, dilation=1)

        # An = b
        matrix_a = point_matrix.view(batch, 3, kernel**2, height, width)  # [B, 3, 9, H, W]
        matrix_a = matrix_a.permute(0, 3, 4, 2, 1) # [B, H, W, 9, 3]
        matrix_a_trans = matrix_a.transpose(3, 4)  # [B, H, W, 3, 9]
        matrix_b = torch.ones([batch, height, width, kernel**2, 1], device=device) #[B, H, W, 9, 1]

        # dot(A.T, A)
        point_multi = torch.matmul(matrix_a_trans, matrix_a) #[B, H, W, 3, 3]
        
        diag = torch.eye(3, device=device).view(1,1,1,3,3) #[1,1,1,3,3]
        diag = diag.repeat(batch,1,1,1,1)           #[B,1,1,3,3]
        # inversible matrix
        try:
            inversible_matrix = point_multi+1e-5*diag
            inv_matrix = torch.inverse(inversible_matrix.to(device))
        except:
            inversible_matrix = point_multi+1e-3*diag
            inv_matrix = torch.inverse(inversible_matrix.to(device))

        # n = (A.T A)^-1 A.T b // || (A.T A)^-1 A.T b ||2
        generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b).squeeze(4) #[B,H,W,3]
        normal = F.normalize(generated_norm, p=2, dim=3)
        # [B,3,H,W]
        normal = normal.permute(0,3,1,2).contiguous()

        replicate_pad = nn.ReplicationPad2d(kernel_1//2)
        normal_pad = replicate_pad(normal)
        # gather normals of the neighbors
        normals = F.unfold(normal_pad, kernel_size=kernel_1, stride=1, padding=0, dilation=1)
        normals = normals.view(batch, 3, kernel_1**2, height, width)  # [B, 3, 25, H, W]

        # d: distance from origin to the plane
        d = torch.sum(normal*points, dim=1, keepdim=True) #[B, 1, H, W]
        d_pad = replicate_pad(d)
        # gather d of the neighbors
        ds = F.unfold(d_pad, kernel_size=kernel_1, stride=1, padding=0, dilation=1)
        ds = ds.view(batch, 1, kernel_1**2, height, width)  # [B, 1, 25, H, W]

        depth_samples = ray_intersection(instrinsic_inv, ds, normals) # [B,1,25,H,W]
        similarity_normal = torch.sum(normal.unsqueeze(2)*normals, dim=1, keepdim=True) # [B,1,25,H,W]
        similarity_normal = torch.exp(similarity_normal-1)
        depth_refined = torch.sum(depth_samples*similarity_normal, dim=2, keepdim=True) / torch.sum(similarity_normal, dim=2, keepdim=True) # [B,1,1,H,W]
        depth_refined = depth_refined.squeeze(1)
        
        depth_refined_clamped = []
        for k in range(batch):
            depth_refined_clamped.append(torch.clamp(depth_refined[k], min=depth_min[k], max=depth_max[k]).unsqueeze(0))
        
        depth_refined = torch.cat(depth_refined_clamped,dim=0)

        del depth_refined_clamped

    # [B,3,H,W] & [B,1,H,W]
    return normal, depth_refined


def ray_intersection(instrinsic_inv, d, normal):
    '''
    intersect the ray from camera origin to pixel with the hypothesis plane
    get the depth of the intersection
    '''
    # instrinsic_inv: [B,3,3]
    # d: [B,1,N,H,W]
    # normal: [B,3,N,H,W]
    batch, _, N, height, width = normal.shape

    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=normal.device),
                            torch.arange(0, width, dtype=torch.float32, device=normal.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        xyz = torch.matmul(instrinsic_inv, xyz) # [B, 3, H*W]
        xyz = xyz.view(batch,3,height,width) # [B,3,H,W]
    
        temp = normal*(xyz.unsqueeze(2)) #[B,3,N,H,W]
        temp = torch.sum(temp,dim=1,keepdim=True)  #[B,1,N,H,W]
        depth_samples = d / (temp+1e-5)
    
    # [B,1,N,H,W]
    return depth_samples

def linear_sampler(correlation, index):
    num_depth = correlation.size()[-2]
    
    xgrid = torch.zeros(index.size(), device="cuda")
    ygrid = 2.0*index/(num_depth-1) - 1
    # print(torch.max[ygrid][0])
    # [B*H*W,2*r+1,1,2]
    grid = torch.cat([xgrid, ygrid], dim=-1)
    # [B*H*W,1,2*r+1,1]

    corr = F.grid_sample(correlation, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)

    return corr

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
    # assert not torch.isnan(inverse_depth).any(), "nan in inverse depth"
    normalized_depth = (inverse_depth - inverse_depth_max) / (inverse_depth_min - inverse_depth_max)
    return normalized_depth

def depth_unnormalization(normalized_depth, inverse_depth_min, inverse_depth_max):
    '''convert the index in inverse range to depth map'''
    inverse_depth = inverse_depth_max + normalized_depth * (inverse_depth_min - inverse_depth_max) # [B,1,H,W]
    depth = 1.0 / inverse_depth
    return depth

# p: probability volume [B, D, H, W]
# depth_values: discrete depth values [B, D]
# get expected value, soft argmin
# return: depth [B, 1, H, W]
def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    depth = depth.unsqueeze(1)
    return depth

def depth_regression_1(p, depth_values):
    depth = torch.sum(p * depth_values, 1)
    depth = depth.unsqueeze(1)
    return depth

def reverse_huber_loss(depth_est,depth_gt):
    absdiff = torch.abs(depth_est-depth_gt)
    C = 0.2 * torch.max(absdiff).item()
    return torch.mean(torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) ))

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def gradient_weight(img):
    img_pad = nn.ReplicationPad2d(padding=(0,1,0,0))(img)
    grad_img_x = torch.abs(img_pad[:, :, :, :-1] - img_pad[:, :, :, 1:])
    img_pad = nn.ReplicationPad2d(padding=(0,0,0,1))(img)
    grad_img_y = torch.abs(img_pad[:, :, :-1, :] - img_pad[:, :, 1:, :])
    grad = torch.cat([grad_img_x, grad_img_y], dim=1)
    grad = torch.norm(grad, p=2, dim=1, keepdim=True)

    return grad

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def MLP(input_dim, output_dim, hidden_dim):
    """ Multi-layer perceptron """
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim[0]))
    layers.append(nn.ReLU(inplace=True))
    
    for i in range(1, len(hidden_dim)):
        layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
        layers.append(nn.ReLU(inplace=True))
    
    layers.append(nn.Linear(hidden_dim[-1], output_dim))
    layers.append(nn.ReLU(inplace=True))
    
    return nn.Sequential(*layers)
