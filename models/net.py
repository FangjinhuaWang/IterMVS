import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .itermvs import *

class FeatureNet(nn.Module):
    def __init__(self, test=False):
        super(FeatureNet, self).__init__()
        self.in_planes = 8
        self.test = test

        self.conv1 = ConvBnReLU(3,8)
        self.layer1 = self._make_layer(16, stride=2)
        self.layer2 = self._make_layer(32, stride=2)
        self.layer3 = self._make_layer(48, stride=2)

        # output convolution
        self.output3 = nn.Conv2d(48, 48, 3, stride=1, padding=1)
        self.output2 = nn.Conv2d(48, 32, 3, stride=1, padding=1)
        self.output1 = nn.Conv2d(48, 16, 3, stride=1, padding=1)
        
        self.inner1 = nn.Conv2d(16, 48, 1, stride=1, padding=0, bias=True)
        self.inner2 = nn.Conv2d(32, 48, 1, stride=1, padding=0, bias=True)
        self.inner3 = nn.Conv2d(48, 48, 1, stride=1, padding=0, bias=True)

    def _make_layer(self, dim, stride=1):   
        layer1 = ResidualBlock(self.in_planes, dim, stride=stride)
        layer2 = ResidualBlock(dim, dim)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.test:
            feas={}
            B,V,_,H,W = x.size()
            x = x.view(B*V,-1,H,W)
            fea0 = self.conv1(x)
            fea1 = self.layer1(fea0)
            fea2 = self.layer2(fea1)
            fea3 = self.layer3(fea2)

            feas["level3"] = torch.unbind(self.output3(fea3).view(B,V,-1,H//8,W//8), dim=1)
            intra_feat = F.interpolate(fea3, scale_factor=2, mode="bilinear") + self.inner2(fea2)
            # del fea2, fea3
            feas['level2'] = torch.unbind(self.output2(intra_feat).view(B,V,-1,H//4,W//4), dim=1)
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear") + self.inner1(fea1)
            # del fea1
            feas['level1'] = torch.unbind(self.output1(intra_feat).view(B,V,-1,H//2,W//2), dim=1)
        else:
            feas={"level3":[],"level2":[],"level1":[]}
            B,V,_,H,W = x.size()
            x = torch.unbind(x, dim=1)
            for i in range(V):
                fea0 = self.conv1(x[i])
                fea1 = self.layer1(fea0)
                fea2 = self.layer2(fea1)
                fea3 = self.layer3(fea2)
                feas["level3"].append(self.output3(fea3))
                intra_feat = F.interpolate(fea3, scale_factor=2, mode="bilinear") + self.inner2(fea2)
                feas["level2"].append(self.output2(intra_feat))
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear") + self.inner1(fea1)
                feas["level1"].append(self.output1(intra_feat))
        return feas

class Pipeline(nn.Module):
    def __init__(self,  iteration=4, test=False):
        super(Pipeline, self).__init__()
        self.feature_dim = [8,16,32,48]
        self.hidden_dim = 32
        self.test = test

        self.feature_net = FeatureNet(test=test)
        self.iter_mvs = IterMVS(iteration, self.feature_dim[2], self.hidden_dim, test)
        
    def forward(self, imgs, proj_matrices, depth_min, depth_max):
        imgs_0 = torch.unbind(imgs['level_0'], 1)
        imgs_2 = torch.unbind(imgs['level_2'], 1)

        features = self.feature_net(imgs['level_0'])
        ref_feature = {
                    "level3":features['level3'][0],
                    "level2":features['level2'][0],
                    "level1":features['level1'][0],
        }
        src_features = {
                    "level3": [src_fea for src_fea in features['level3'][1:]],
                    "level2": [src_fea for src_fea in features['level2'][1:]],
                    "level1": [src_fea for src_fea in features['level1'][1:]],
        }

        proj_matrices_1 = torch.unbind(proj_matrices['level_1'].float(), 1)
        proj_matrices_2 = torch.unbind(proj_matrices['level_2'].float(), 1)
        proj_matrices_3 = torch.unbind(proj_matrices['level_3'].float(), 1)
        ref_proj = {
                "level3": proj_matrices_3[0],
                "level2": proj_matrices_2[0],
                "level1": proj_matrices_1[0]
        }
        src_projs = {
                "level3": proj_matrices_3[1:],
                "level2": proj_matrices_2[1:],
                "level1": proj_matrices_1[1:]
        }
        
        depth_min = depth_min.float()
        depth_max = depth_max.float()

        if not self.test:
            depths, depths_upsampled, confidences, confidence_upsampled = self.iter_mvs(ref_feature, src_features,
                        ref_proj, src_projs, depth_min, depth_max)

            return {
                        "depths": depths, 
                        "depths_upsampled": depths_upsampled,
                        "confidences": confidences,
                        "confidence_upsampled": confidence_upsampled,
                    }
        else:
            depth, depths_upsampled, confidence, confidence_upsampled = self.iter_mvs(ref_feature, src_features,
                        ref_proj, src_projs, depth_min, depth_max)

            return {
                        "depths_upsampled": depths_upsampled,
                        "confidence_upsampled": confidence_upsampled,
                    }


def full_loss(depths, depths_upsampled, confidences, depths_gt, mask, depth_min, depth_max, regress=True):  
    loss = 0
    radius = 4
    out_num_samples = 256
    cross_entropy = nn.BCEWithLogitsLoss()
    depth_probability = depths["probability"]
    num_sample = depth_probability[0].size(1)

    mask_0 = mask['level_0'] > 0.5
    mask_1 = mask['level_2'] > 0.5
    depth_gt_0 = depths_gt['level_0']
    depth_gt_1 = depths_gt['level_2']


    batch, _, height, width = depth_gt_1.size()
    inverse_depth_min = (1.0 / depth_min).view(batch,1,1,1)
    inverse_depth_max = (1.0 / depth_max).view(batch,1,1,1)
    normalized_depth_gt = depth_normalization(depth_gt_1, inverse_depth_min, inverse_depth_max)

    gt_index =  torch.clamp(normalized_depth_gt,min=0,max=1) * (num_sample-1)
    gt_index = gt_index * mask_1.float()
    gt_index = torch.floor(gt_index).type(torch.long)
    gt_probability = torch.zeros_like(depth_probability[0]).scatter_(1, gt_index, 1)

    num_prediction = len(depths["combine"])
    
    coff = 0.8 ** num_prediction
    depth = depths["initial"][0]
    normalized_depth = depth_normalization(depth, inverse_depth_min, inverse_depth_max)
    loss = loss + coff * out_num_samples * F.l1_loss(normalized_depth[mask_1], normalized_depth_gt[mask_1], reduction='mean')
    

    for iter in range(num_prediction):
        coff = 0.8 ** (num_prediction-iter-1)
        probability = depth_probability[iter]
        probability = torch.clamp(probability, min=1e-5)
        loss_probability = - torch.sum(gt_probability * torch.log(probability), dim=1, keepdim=True)
        loss = loss + coff * torch.mean(loss_probability[mask_1])

        if regress:
            with torch.no_grad():
                index = torch.argmax(probability, dim=1, keepdim=True).type(torch.float)
                mask_2 = (gt_index >= index-radius) & (gt_index <= index+radius)
            depth = depths["combine"][iter]
            normalized_depth = depth_normalization(depth, inverse_depth_min, inverse_depth_max)
            mask_new = mask_1 & mask_2
            if torch.sum(mask_new)>0:
                loss = loss + coff * out_num_samples * F.l1_loss(normalized_depth[mask_new], normalized_depth_gt[mask_new], reduction='mean')

            confidence = confidences[iter]
            confidence_masked = confidence[mask_1]
            confidence_gt = (torch.abs(normalized_depth[mask_1].detach() - normalized_depth_gt[mask_1]) < 0.002).float()
            loss = loss + coff * cross_entropy(confidence_masked, confidence_gt)

    normalized_depth_gt = depth_normalization(depth_gt_0, inverse_depth_min, inverse_depth_max)
    depth = depths_upsampled[0]
    normalized_depth = depth_normalization(depth, inverse_depth_min, inverse_depth_max)
    loss = loss + out_num_samples * F.l1_loss(normalized_depth[mask_0], normalized_depth_gt[mask_0], reduction='mean')

    return loss
