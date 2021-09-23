from numpy import imag
from numpy.lib.function_base import diff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .module import *
from .patchmatch import *
import cv2

'''
class FeatureNet(nn.Module):
    def __init__(self, output_dim=128):
        super(FeatureNet, self).__init__()
        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        # [B,8,H,W]
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)
        # [B,16,H/2,W/2]
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)
        # [B,32,H/4,W/4]
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv7 = ConvBnReLU(32, 32, 3, 1, 1)
        # [B,64,H/8,W/8]
        self.conv8 = ConvBnReLU(32, 64, 5, 2, 2)
        self.conv9 = ConvBnReLU(64, 64, 3, 1, 1)
        self.conv10 = ConvBnReLU(64, 64, 3, 1, 1)
        
        self.inner = nn.Conv2d(32, 64, 1, bias=True)
        self.output = nn.Conv2d(64, output_dim, 1, bias=False)
        
        
     
    def forward(self, x):
        conv = self.conv1(self.conv0(x))
        conv = self.conv4(self.conv3(self.conv2(conv)))
        conv7 = self.conv7(self.conv6(self.conv5(conv)))
        del conv
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))
        
        intra_feat = F.interpolate(conv10, scale_factor=2, mode="bilinear") + self.inner(conv7)
        
        return self.output(intra_feat)
        
        
'''

class FeatureNet(nn.Module):
    def __init__(self, dropout=0.0, test=False):
        super(FeatureNet, self).__init__()
        self.in_planes = 8
        self.test = test
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3,8,3,1,1),
        #     nn.InstanceNorm2d(8),
        #     nn.ReLU(inplace=True)
        # )
        self.conv1 = ConvBnReLU(3,8)
        self.layer1 = self._make_layer(16, stride=2)
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(8,16,3,2,1),
        #     nn.InstanceNorm2d(16),
        #     nn.ReLU(inplace=True)
        # )
        self.layer2 = self._make_layer(32, stride=2)
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16,32,3,2,1),
        #     nn.InstanceNorm2d(32),
        #     nn.ReLU(inplace=True)
        # )
        self.layer3 = self._make_layer(48, stride=2)
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(32,48,3,2,1),
        #     nn.InstanceNorm2d(48),
        #     nn.ReLU(inplace=True)
        # )
        # self.layer4 = self._make_layer(64, stride=2)

        # output convolution
        # self.output4 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.output3 = nn.Conv2d(48, 48, 3, stride=1, padding=1)
        self.output2 = nn.Conv2d(48, 32, 3, stride=1, padding=1)
        self.output1 = nn.Conv2d(48, 16, 3, stride=1, padding=1)
        
        self.inner1 = nn.Conv2d(16, 48, 1, stride=1, padding=0, bias=True)
        self.inner2 = nn.Conv2d(32, 48, 1, stride=1, padding=0, bias=True)
        self.inner3 = nn.Conv2d(48, 48, 1, stride=1, padding=0, bias=True)
        # if not self.test:
        #     self.vgg = models.vgg16(pretrained=True)
        #     for p in self.vgg.parameters():
        #         p.requires_grad=False

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
    def _make_layer(self, dim, stride=1):   
        layer1 = ResidualBlock(self.in_planes, dim, stride=stride)
        layer2 = ResidualBlock(dim, dim)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if not self.test:
        #     feas={}
        #     B,V,_,H,W = x.size()
        #     x = x.view(B*V,-1,H,W)
        #     fea0 = self.conv1(x)
        #     fea1 = self.layer1(fea0)
        #     fea2 = self.layer2(fea1)
        #     fea3 = self.layer3(fea2)

        #     feas["level3"] = torch.unbind(self.output3(fea3).view(B,V,-1,H//8,W//8), dim=1)
        #     intra_feat = F.interpolate(fea3, scale_factor=2, mode="bilinear") + self.inner2(fea2)
        #     # del fea2, fea3
        #     feas['level2'] = torch.unbind(self.output2(intra_feat).view(B,V,-1,H//4,W//4), dim=1)
        #     intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear") + self.inner1(fea1)
        #     # del fea1
        #     feas['level1'] = torch.unbind(self.output1(intra_feat).view(B,V,-1,H//2,W//2), dim=1)
        # else:
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
    def __init__(self,  iteration = 5, height0=512, width0=640, test=False):
        super(Pipeline, self).__init__()
        self.feature_dim = [8,16,32,48]
        self.context_dim = 32
        self.hidden_dim = 32
        self.test = test
        
        self.feature_net = FeatureNet(test=test)
        # self.position_encoding = MLP(2, self.feature_dim, [8,32,64])
        # self.context_net = FeatureNet(output_dim=self.context_dim + self.hidden_dim)
        # self.context_net = FeatureNet(output_dim=self.context_dim)
        self.iter_mvs = IterMVS(iteration, self.feature_dim[2], self.hidden_dim, self.context_dim, height0, width0, test)
        # self.refinement = Refinement()
        
    def forward(self, imgs, proj_matrices, relative_extrinsic, intrinsics_inv, depth_min, depth_max):
        imgs_0 = torch.unbind(imgs['stage_0'], 1)
        imgs_2 = torch.unbind(imgs['stage_2'], 1)
        # imgs_2_ref = imgs_2[0]
        # imgs_2_src = imgs_2[1:]
        # del imgs
        # batch, _, height, width = imgs_2[0].size()
        # device = imgs_2[0].device


        # step 1. Multi-scale feature extraction
        # for img in imgs_0:
        features = self.feature_net(imgs['stage_0'])

        '''
        pos_cod = {"level1": positionalencoding2d(self.feature_dim[1], height*2, width*2, device),
                    "level2": positionalencoding2d(self.feature_dim[2], height, width, device),
                    "level3": positionalencoding2d(self.feature_dim[3], height//2, width//2, device),}
        
        ref_feature = {"level3":features[0]['level3']+pos_cod["level3"],
                        "level2":features[0]['level2']+pos_cod["level2"],
                        "level1":features[0]['level1']+pos_cod["level1"],}
        
        src_features = {"level3": [src_fea['level3']+pos_cod["level3"] for src_fea in features[1:]],
                        "level2": [src_fea['level2']+pos_cod["level2"] for src_fea in features[1:]],
                        "level1": [src_fea['level1']+pos_cod["level1"] for src_fea in features[1:]],}
        '''
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

        # relative_extrinsic = torch.unbind(relative_extrinsic.float(), 1)
        
        proj_matrices_1 = torch.unbind(proj_matrices['stage_1'].float(), 1)
        proj_matrices_2 = torch.unbind(proj_matrices['stage_2'].float(), 1)
        proj_matrices_3 = torch.unbind(proj_matrices['stage_3'].float(), 1)
        assert len(imgs_0) == len(proj_matrices_1), "Different number of images and projection matrices"

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
        # del proj_matrices, proj_matrices_1, proj_matrices_2, proj_matrices_3
        intrinsics_inv = intrinsics_inv['stage_2']
        
        depth_min = depth_min.float()
        depth_max = depth_max.float()

        
        if not self.test:
            depths, depths_upsampled, confidences, confidence_upsampled = self.iter_mvs(ref_feature, src_features,
                        ref_proj, src_projs, intrinsics_inv, depth_min, depth_max, imgs_2[0], imgs_2[1:])
            
            return {
                        "depths": depths, 
                        "confidences": confidences,
                        # "view_weights": view_weights,
                        # "normals": normals,
                        "depths_upsampled": depths_upsampled,
                        "confidence_upsampled": confidence_upsampled,
                        "features": [ref_feature, src_features],
                        # "view_weights": view_weights,
                    }
        else:
            depth, depths_upsampled, confidence, confidence_upsampled = self.iter_mvs(ref_feature, src_features,
                        ref_proj, src_projs, intrinsics_inv, depth_min, depth_max, imgs_2[0], imgs_2[1:])
            
            return {
                        "depths": depth, 
                        "depths_upsampled": depths_upsampled,
                        "confidences": confidence,
                        "confidence_upsampled": confidence_upsampled,
                    }
            




def full_loss(depths, depths_upsampled, confidences, depths_gt, mask, depth_min, depth_max, imgs, proj_matrices, regress=True):  
    loss = 0
    radius = 4
    out_num_samples = 256
    cross_entropy = nn.BCEWithLogitsLoss()
    depth_probability = depths["probability"]
    num_sample = depth_probability[0].size(1)

    mask_0 = mask['stage_0'] > 0.5
    mask_1 = mask['stage_2'] > 0.5
    depth_gt_0 = depths_gt['stage_0']
    depth_gt_1 = depths_gt['stage_2']
    ref_img_0 = torch.unbind(imgs['stage_0'].float(), 1)[0]
    ref_img_2 = torch.unbind(imgs['stage_2'].float(), 1)[0]

    batch, _, height, width = depth_gt_1.size()
    inverse_depth_min = (1.0 / depth_min).view(batch,1,1,1)
    inverse_depth_max = (1.0 / depth_max).view(batch,1,1,1)
    normalized_depth_gt = depth_normalization(depth_gt_1, inverse_depth_min, inverse_depth_max)

    gt_index =  torch.clamp(normalized_depth_gt,min=0,max=1) * (num_sample-1)
    gt_index = gt_index * mask_1.float()
    gt_index = torch.floor(gt_index).type(torch.long)
    gt_probability = torch.zeros_like(depth_probability[0]).scatter_(1, gt_index, 1)

    num_prediction = len(depths["combine"])

    # if mask_1[0,0,height//2,width//2] == 1:
    #     print("gt:{:.2f}".format(depth_gt_1[0,0,height//2,width//2]))
    
    depth_gt_masked = depth_gt_1[mask_1]

    """
    coff = 0.8 ** num_prediction
    depth = depths["initial"][0]
    # normalized_depth = depth_normalization(depth, inverse_depth_min, inverse_depth_max)
    loss = loss + coff * F.l1_loss(depth[mask_1], depth_gt_masked, reduction='mean')
    # loss = loss + coff * out_num_samples * F.l1_loss(normalized_depth[mask_1], normalized_depth_gt[mask_1], reduction='mean')
    """

    for iter in range(num_prediction):
        coff = 0.8 ** (num_prediction-iter-1)
        probability = depth_probability[iter]
        probability = torch.clamp(probability, min=1e-5)
        loss_probability = - torch.sum(gt_probability * torch.log(probability), dim=1, keepdim=True)
        loss = loss + coff * torch.mean(loss_probability[mask_1])
        # grad_weight = gradient_weight(ref_img_2)
        confidence = confidences[iter]
        confidence_masked = confidence[mask_1]
        depth = depths["combine"][iter]
        normalized_depth = depth_normalization(depth, inverse_depth_min, inverse_depth_max)
        confidence_gt = (torch.abs(normalized_depth[mask_1].detach() - normalized_depth_gt[mask_1]) < 0.002).float()
        loss = loss + coff * cross_entropy(confidence_masked, confidence_gt)

        if regress:
            with torch.no_grad():
                index = torch.argmax(probability, dim=1, keepdim=True).type(torch.float)
                mask_2 = (gt_index >= index-radius) & (gt_index <= index+radius)
            # depth = depths["combine"][iter]
            # normalized_depth = depth_normalization(depth, inverse_depth_min, inverse_depth_max)
            mask_new = mask_1 & mask_2
            if torch.sum(mask_new)>0:
                # depth_masked = depth[mask_new]
                # loss = loss + coff * torch.mean((grad_weight*torch.abs(depth-depth_gt_1))[mask_new])
                loss = loss + coff * F.l1_loss(depth[mask_new], depth_gt_1[mask_new], reduction='mean')
                # loss = loss + coff * out_num_samples * F.l1_loss(normalized_depth[mask_new], normalized_depth_gt[mask_new], reduction='mean')

                # confidence = confidences[iter]
                # confidence_masked = confidence[mask_new]
                # confidence_gt = (torch.abs(normalized_depth[mask_new].detach() - normalized_depth_gt[mask_new]) < 0.002).float()
                # loss = loss + coff * cross_entropy(confidence_masked, confidence_gt)


        '''
        if regress:
            depth = depths["combine"][iter]
            normalized_depth = depth_normalization(depth, inverse_depth_min, inverse_depth_max)
            depth_masked = depth[mask_1]
            loss = loss + coff * F.l1_loss(depth_masked, depth_gt_masked, reduction='mean')
            
            confidence = confidences[iter]
            confidence_masked = confidence[mask_1]
            confidence_gt = (torch.abs(normalized_depth[mask_1].detach() - normalized_depth_gt[mask_1]) < 0.002).float()
            loss = loss + coff * cross_entropy(confidence_masked, confidence_gt)
        '''
    
    # normalized_depth_gt = depth_normalization(depth_gt_0, inverse_depth_min, inverse_depth_max)
    # if regress:
    # grad_weight = gradient_weight(ref_img_0)
    depth = depths_upsampled[0]
    # normalized_depth = depth_normalization(depth, inverse_depth_min, inverse_depth_max)
    # depth_masked = depth[mask_0]
    # depth_gt_masked = depth_gt_0[mask_0]
    # loss = loss + torch.mean((grad_weight*torch.abs(depth-depth_gt_0))[mask_0])
    loss = loss + F.l1_loss(depth[mask_0], depth_gt_0[mask_0], reduction='mean')
    # loss = loss + out_num_samples * F.l1_loss(normalized_depth[mask_0], normalized_depth_gt[mask_0], reduction='mean')
    
    return loss

def image_loss(imgs, proj_matrices, depth):
    ref_proj = proj_matrices[0]
    src_projs = proj_matrices[1:]
    ref_img = imgs[0]
    src_imgs = imgs[1:]

    loss = 0.0
    for src_img, src_proj in zip(src_imgs, src_projs):
        warped_img, mask = differentiable_warping(src_img,src_proj,ref_proj,depth, True)
        warped_img = warped_img.squeeze(2)
        loss = loss + F.l1_loss(ref_img[mask], warped_img[mask], reduction="mean")

    return loss

def feature_loss(ref_feature, src_features, proj_matrices, depth):
    ref_proj = proj_matrices[0]
    src_projs = proj_matrices[1:]

    loss = 0.0
    for src_feature, src_proj in zip(src_features, src_projs):
        warped_fea, mask = differentiable_warping(src_feature,src_proj,ref_proj,depth, True)
        warped_fea = warped_fea.squeeze(2)
        # loss = loss + F.l1_loss(ref_feature[mask], warped_fea[mask], reduction="mean")
        fea_diff = torch.sum(torch.abs(ref_feature-warped_fea), dim=1, keepdim=True)
        # print(fea_diff.size())
        loss = loss + torch.mean(fea_diff[mask])

    return loss

def flow_loss(proj_matrices, depth, depth_gt, mask):
    ref_proj = proj_matrices[0]
    src_projs = proj_matrices[1:]
    loss = 0.0

    for src_proj in src_projs:
        proj_xy_gt, valid_mask = differentiable_warping_grid(src_proj, ref_proj, depth_gt)
        proj_xy, _ = differentiable_warping_grid(src_proj, ref_proj, depth)
        # print(proj_xy)
        valid_mask = valid_mask & mask
        # print(torch.sum(valid_mask))
        flow_diff = torch.sum(torch.abs(proj_xy_gt - proj_xy), dim=1,keepdim=True)
        if torch.sum(valid_mask)>0:
            loss = loss + torch.mean(flow_diff[valid_mask])
        # loss = loss + F.l1_loss(proj_xy_gt[valid_mask], proj_xy[valid_mask], reduction='mean')
    
    return loss