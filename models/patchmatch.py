from typing import BinaryIO
from numpy.core.fromnumeric import std
import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from .module import *
import cv2
import numpy as np
# import torchgeometry as tgm
from copy import deepcopy

class DepthInitialization(nn.Module):
    def __init__(self, num_sample):
        super(DepthInitialization, self).__init__()
        self.num_sample = num_sample
    
    def forward(self, inverse_depth_min, inverse_depth_max, height, width, device):
        batch = inverse_depth_min.size()[0]
        # first iteration of Patchmatch, uniformly sample in the inverse depth range
        # hypothesis is front-to-parallel
        
        index = torch.arange(0, self.num_sample, 1, device=device).view(1, self.num_sample, 1, 1).float()
        
        normalized_sample = index.repeat(batch, 1, height, width) / (self.num_sample-1)
       
        depth_sample = inverse_depth_max + normalized_sample * (inverse_depth_min - inverse_depth_max)
        
        depth_sample = 1.0 / depth_sample
        

        return depth_sample
        

class Evaluation(nn.Module):
    '''
    compute the correlation of all depth samples for each pixel
    '''
    def __init__(self, height, width):
        super(Evaluation, self).__init__()
        self.G = 8
        self.pixel_view_weight = PixelViewWeight(self.G, height, width)
        # self.feature_conv = nn.Conv2d(48+32+16,48,3,1,1)
        self.corr_conv1 = nn.ModuleList([CorrNet(self.G), CorrNet(self.G), CorrNet(self.G)])


    def forward(self, ref_feature, src_features, ref_proj, src_projs, depth_sample, inverse_depth_min=None, inverse_depth_max=None, view_weights=None):
        V = len(src_features["level2"])+1
        # device = ref_feature["level2"].get_device()

        if view_weights == None:
            correlation_sum = 0
            view_weight_sum = 1e-5
            view_weights = []
            batch, dim, height, width = ref_feature["level3"].size()
            
            ref_feature = ref_feature["level3"]
            src_features = src_features["level3"]

            ref_proj = ref_proj["level3"]
            num_sample = depth_sample.size()[1]

            for src_feature, src_proj in zip(src_features, src_projs["level3"]):
                warped_feature = differentiable_warping(src_feature, src_proj, ref_proj, depth_sample) #[B,C,N,H,W]
                warped_feature = warped_feature.view(batch, self.G, dim//self.G, num_sample, height, width)
                correlation = torch.mean(warped_feature*ref_feature.view(batch, self.G, dim//self.G, 1, height, width), dim=2, keepdim=False) #[B,G,N,H,W]
                # correlation = torch.sum(warped_feature*ref_feature.view(batch, dim, 1, height, width), dim=1, keepdim=False)/torch.sqrt(torch.tensor(dim).float())
                view_weight = self.pixel_view_weight(correlation) # [B,1,H,W]
                # correlation = torch.mean(correlation, dim=1, keepdim=False)
                
                # correlation = self.corr_conv(correlation).view(batch, num_sample, height, width)
                
                del warped_feature, src_feature, src_proj
                view_weights.append(F.interpolate(view_weight,
                                    scale_factor=2, mode='bilinear'))
                # view_angle_weight = ViewAngleWeight(relative_extrinsic, intrinsics_inv, depth_sample)
                # view_weight = view_weight * view_angle_weight
                
                if self.training:
                    correlation_sum = correlation_sum + correlation * view_weight.unsqueeze(1) # [B, N, H, W]
                    view_weight_sum = view_weight_sum + view_weight.unsqueeze(1)  #[B,N,H,W]
                else:
                    correlation_sum += correlation * view_weight.unsqueeze(1)
                    view_weight_sum += view_weight.unsqueeze(1)
                '''
                # visualization of pixel-wise view weight
                view_weight = view_weight.permute(0,2,3,1).contiguous().view(-1,height,width,1)
                view_weight = view_weight[0].detach().cpu().numpy()
                # view_weight = view_weight*4
                view_weight=np.clip(view_weight,0,1)
                view_weight = (view_weight*255).astype(np.uint8)
                view_weight_img = cv2.applyColorMap(view_weight,2)

                cv2.imshow(f'view_weight_source_image', view_weight_img)
                cv2.waitKey(1)
                # global index
                # cv2.imwrite("/home/fangjinhuawang/Desktop/mvs/mvs_115/pixel_wise_weight/{}.png".format(index), view_weight_img)
                # index=index+1
                '''
                del correlation, view_weight
            del src_features, src_projs

            # aggregated matching cost across all the source views
            correlation = correlation_sum.div_(view_weight_sum) # [B,G,N,H,W]
            correlation = self.corr_conv1[-1](correlation)
            view_weights = torch.cat(view_weights,dim=1)#.detach()
            del correlation_sum, view_weight_sum
            
            probability = torch.softmax(correlation, dim=1)
            # del correlation
            index = torch.arange(0, num_sample, 1, device=correlation.device).view(1, num_sample, 1, 1).float()
            index = torch.sum(index * probability, dim = 1, keepdim=True) # [B,1,H,W]
            normalized_depth = index / (num_sample-1.0)
            depth = depth_unnormalization(normalized_depth, inverse_depth_min, inverse_depth_max)
            depth = F.interpolate(depth,
                                    scale_factor=2, mode='bilinear')
            '''
            with torch.no_grad():
                probability_sum = 4 * F.avg_pool3d(F.pad(probability.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
                del probability
                # [B, 1, H, W]
                index = index.long()
                confidence = torch.gather(probability_sum, 1, index) # [B,1,H,W]
                del probability_sum
            confidence = F.interpolate(confidence,
                                    scale_factor=2, mode='nearest')
            '''
            return view_weights, correlation, depth

        else:
            correlations = []

            for l in range(1,4):
                # print(l)
                correlation_sum = 0
                view_weight_sum = 1e-5
                ref_feature_l = ref_feature[f"level{l}"]
                ref_proj_l = ref_proj[f"level{l}"]
                depth_sample_l = depth_sample[f"level{l}"]
                batch, num_sample, height, width = depth_sample_l.size()
                dim = ref_feature_l.size(1)

                if not l==2:
                    # need to interpolate
                    ref_feature_l = F.interpolate(ref_feature_l,
                                    scale_factor=2**(l-2), mode='bilinear')

                i=0
                for src_feature, src_proj in zip(src_features[f"level{l}"], src_projs[f"level{l}"]):
                    warped_feature = differentiable_warping(src_feature, src_proj, ref_proj_l, depth_sample_l) #[B,C,N,H,W]
                    warped_feature = warped_feature.view(batch, self.G, dim//self.G, num_sample, height, width)
                    correlation = torch.mean(warped_feature*ref_feature_l.view(batch, self.G, dim//self.G, 1, height, width), dim=2, keepdim=False) #[B,G,N,H,W]
                    view_weight = view_weights[:,i].view(batch,1,1,height,width) #[B,1,H,W]

                    i=i+1
                    del warped_feature, src_feature, src_proj

                    if self.training:
                        correlation_sum = correlation_sum + correlation * view_weight  # [B, N, H, W]
                        # cost_sum = cost_sum + cost * view_weight
                        view_weight_sum = view_weight_sum + view_weight  #[B,1,H,W]

                    else:
                        correlation_sum += correlation * view_weight
                        # cost_sum += cost * view_weight
                        view_weight_sum += view_weight
                    del correlation
                # del src_features, src_projs

                # aggregated matching cost across all the source views
                correlation = correlation_sum.div_(view_weight_sum) # [B,G,N,H,W]
                # correlation = correlation.permute(0,2,1,3,4).contiguous().view(batch*num_sample, self.G, height, width) # [B*N,G,H,W]
                correlation = self.corr_conv1[l-1](correlation)
                correlations.append(correlation)
                # costs.append(cost)
                # correlations[f"level{l}"]=correlation
                del correlation_sum, correlation
            
            correlations = torch.cat(correlations, dim=1)

            return correlations


class Update(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_sample):
        super(Update, self).__init__()
        self.G = 4
        self.hidden_dim = hidden_dim
        # self.gru = SepConvGRU(hidden_dim, input_dim)
        self.gru = ConvGRU(hidden_dim, input_dim)
        # self.lstm = ConvLSTM(hidden_dim, input_dim)
        self.out_num_samples = 256
        self.radius = 4
        self.depth_head = nn.Sequential(
                nn.Conv2d(hidden_dim, 32, 3, stride=1, padding=2, dilation=2,bias=False), 
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 1, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, self.out_num_samples, 1, stride=1, padding=0, dilation=1),
                )

        self.confidence_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, stride=1, padding=2, dilation=2,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, stride=1, padding=0, dilation=1),
            )

        self.hidden_init_head = nn.Sequential(
            nn.Conv2d(num_sample, 64, 3, stride=1, padding=1, dilation=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, 1, stride=1, padding=0, dilation=1),
        )
        
    def hidden_init(self, corr):
        hidden = self.hidden_init_head(corr)
        hidden = F.interpolate(hidden,
                        scale_factor=2, mode='bilinear')
        hidden = torch.tanh(hidden)
        return hidden

    def conf_init(self, hidden):
        confidence_0 = self.confidence_head(hidden)
        confidence = torch.sigmoid(confidence_0)
        return confidence, confidence_0
    
    def depth_init(self, hidden):
        probability = torch.softmax(self.depth_head(hidden), dim=1)
        with torch.no_grad():
            index = torch.argmax(probability, dim=1, keepdim=True).type(torch.float)
            index_low = index - self.radius
            index = torch.arange(0, 2*self.radius+1, 1, device=hidden.device).view(1, 2*self.radius+1, 1, 1).float()
            index = index_low+index
            index = torch.clamp(index, min=0, max=self.out_num_samples-1)
            index = index.type(torch.long)

        # height, width=probability.size()[2:]
        regress_index = 0
        probability_sum = 1e-6
        for i in range(2*self.radius+1):
            probability_1 = torch.gather(probability, 1, index[:,i:i+1])
            # print("i:{}, proba:{:.5f}".format(i, probability_1[0,0,height//2,width//2]))
            regress_index = regress_index + index[:,i:i+1]*probability_1
            probability_sum = probability_sum+probability_1
        regress_index = regress_index.div_(probability_sum)

        # print("index:{}, regres:{:.2f}".format(index[0,self.radius,height//2,width//2], regress_index[0,0,height//2,width//2]))
        normalized_depth = regress_index / (self.out_num_samples-1.0)
        return normalized_depth, probability

    def forward(self, hidden, normalized_depth, corr, context=None, confidence=None, confidence_flag=False):
        cat = torch.cat([normalized_depth, corr], dim=1)        
        hidden = self.gru(hidden, cat, confidence)
        confidence_new_0 = None
        confidence_new = None
        if confidence_flag:
            confidence_new_0 = self.confidence_head(hidden)
            confidence_new = torch.sigmoid(confidence_new_0)
        
        probability = torch.softmax(self.depth_head(hidden), dim=1)

        with torch.no_grad():
            index = torch.argmax(probability, dim=1, keepdim=True).type(torch.float)
            index_low = index - self.radius
            index = torch.arange(0, 2*self.radius+1, 1, device=hidden.device).view(1, 2*self.radius+1, 1, 1).float()
            index = index_low+index
            index = torch.clamp(index, min=0, max=self.out_num_samples-1)
            index = index.type(torch.long)

        regress_index = 0
        probability_sum = 1e-6
        for i in range(2*self.radius+1):
            probability_1 = torch.gather(probability, 1, index[:,i:i+1])
            # print("i:{}, proba:{:.5f}".format(i, probability_1[0,0,height//2,width//2]))
            regress_index = regress_index + index[:,i:i+1]*probability_1
            probability_sum = probability_sum+probability_1
        regress_index = regress_index.div_(probability_sum)

        # print("index:{}, regres:{:.2f}".format(index[0,self.radius,height//2,width//2], regress_index[0,0,height//2,width//2]))
        normalized_depth = regress_index / (self.out_num_samples-1.0)
        
        return hidden, normalized_depth, probability, confidence_new, confidence_new_0
        



class IterMVS(nn.Module):
    def __init__(self, iteration, feature_dim, hidden_dim, context_dim=None, height0=512, width0=640, test=False):
        super(IterMVS, self).__init__()
        self.iteration = iteration
        self.hidden_dim = hidden_dim
        # self.context_dim = context_dim
        self.corr_sample = 12
        self.interval_max = 0.01
        self.interval_min = 0.001
        self.interval = 1.0/256 #0.005


        self.corr_interval = {
                                # "level1": torch.FloatTensor([-2,-2.0/3,2.0/3,2]).view(1,4,1,1),
                                # "level2": torch.FloatTensor([-16,-16.0/3,16.0/3,16]).view(1,4,1,1),
                                # "level3": torch.FloatTensor([-64,-64.0/3,64.0/3,64]).view(1,4,1,1)
                                "level1": torch.FloatTensor([-2,-2.0/3,2.0/3,2]).view(1,4,1,1),
                                "level2": torch.FloatTensor([-8,-8.0/3,8.0/3,8]).view(1,4,1,1),
                                "level3": torch.FloatTensor([-32,32]).view(1,2,1,1)
                            }

        self.num_sample = 36
        self.test = test

        # self.depth_initialization = DepthInitialization(feature_dim)
        self.depth_initialization = DepthInitialization(self.num_sample)
        # self.propagation = Propagation()
        self.evaluation = Evaluation(height0//4, width0//4)
        
        self.update = Update(1 + 10, hidden_dim, self.num_sample)

        self.upsample = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16*9, 1, stride=1, padding=0, dilation=1, bias=False)
        )

    
    def forward(self, ref_feature, src_features, ref_proj, src_projs, intrinsics_inv, depth_min, depth_max, img = None, src_imgs=None):
        depths = {"combine":[],"probability":[], "initial":[]}
        confidences = []
        depths_upsampled = []
        # confidences_upsampled = []
        confidence_upsampled = None
        
        device = ref_feature["level2"].device
        batch, _, height, width = ref_feature["level2"].size()

        upsample_weight = self.upsample(ref_feature["level2"]) # [B,16*9,H,W]
        upsample_weight = upsample_weight.view(batch,1,9,4,4,height,width)
        upsample_weight = torch.softmax(upsample_weight, dim=2)


        inverse_depth_min = (1.0 / depth_min).view(batch,1,1,1)
        inverse_depth_max = (1.0 / depth_max).view(batch,1,1,1)

        depth_samples = self.depth_initialization(inverse_depth_min, inverse_depth_max, height//2, width//2, device)
        view_weights, corr, depth = self.evaluation(ref_feature, src_features, ref_proj, src_projs, depth_samples, inverse_depth_min, inverse_depth_max) # [B,2r+1,H,W]
        if not self.test:
            depths["initial"].append(depth)

        # initialization only gives correlation used to initialize hidden
        hidden = self.update.hidden_init(corr)            
        normalized_depth, probability = self.update.depth_init(hidden)
        print_sample = ""

        if not self.test:
            confidence, confidence_0 = self.update.conf_init(hidden)
            depth = depth_unnormalization(normalized_depth, inverse_depth_min, inverse_depth_max)
            depths["combine"].append(depth)
            depths["probability"].append(probability)

            confidences.append(confidence_0)
            confidence = confidence.detach()
            normalized_depth = normalized_depth.detach()

            print_sample += "depth {:.2f}, conf:{:.2f} ".format(depth[0,0,height//2, width//2].item(), confidence[0,0,height//2, width//2].item())


        for iter in range(self.iteration):
            samples = {}
            for i in range(1,4):
                normalized_sample = normalized_depth + self.corr_interval[f"level{i}"].to(device) * self.interval # [B,R,H,W] 
                normalized_sample = torch.clamp(normalized_sample, min=0, max=1)
                samples[f"level{i}"] = depth_unnormalization(normalized_sample, inverse_depth_min, inverse_depth_max)

            corr = self.evaluation(ref_feature, src_features, ref_proj, src_projs, samples, view_weights=view_weights.detach()) # [B,2r+1,H,W]

            if not self.test:
                hidden, normalized_depth, probability, confidence, confidence_0 = self.update(hidden, normalized_depth, corr, context=img, confidence_flag=True)

                depth = depth_unnormalization(normalized_depth, inverse_depth_min, inverse_depth_max)
                depths["combine"].append(depth)
                depths["probability"].append(probability)

                confidences.append(confidence_0)

                print_sample += "depth {:.2f}, conf:{:.2f}".format(depth[0,0,height//2, width//2].item(), confidence[0,0,height//2, width//2].item())

                if iter==self.iteration-1:
                    depth_upsampled = upsample(normalized_depth, upsample_weight)
                    depth_upsampled = depth_unnormalization(depth_upsampled, inverse_depth_min, inverse_depth_max)
                    depths_upsampled.append(depth_upsampled)
                    # confidence_upsampled = upsample(confidence, upsample_weight_conf)
                    confidence_upsampled = F.interpolate(confidence,
                                            scale_factor=4, mode='bilinear')
                
                confidence = confidence.detach()
                normalized_depth = normalized_depth.detach()
            else:
                if iter<self.iteration-1:
                    hidden, normalized_depth, _, _, _ = self.update(hidden, normalized_depth, corr, context=img, confidence_flag=False)
                else:
                    depth = depth_unnormalization(normalized_depth, inverse_depth_min, inverse_depth_max)
                    hidden, normalized_depth, _, confidence, _ = self.update(hidden, normalized_depth, corr, context=img, confidence_flag=True)
                    depth_upsampled = upsample(normalized_depth, upsample_weight)
                    depth_upsampled = depth_unnormalization(depth_upsampled, inverse_depth_min, inverse_depth_max)
                    confidence_upsampled = F.interpolate(confidence,
                                            scale_factor=4, mode='bilinear')
                    # confidence_upsampled = upsample(confidence, upsample_weight_conf)

        if self.test:
            return depth, depth_upsampled, confidence, confidence_upsampled
        else:
            # print(print_sample)
            return depths, depths_upsampled, confidences, confidence_upsampled


# estimate pixel-wise view weight
class PixelViewWeight(nn.Module):
    def __init__(self, G, height, width):
        super(PixelViewWeight, self).__init__()

        self.conv = nn.Sequential(
            ConvReLU(G, 16),
            nn.Conv2d(16, 1, 1, stride=1, padding=0),
        )


    def forward(self, x):
        # x: [B, G, N, H, W]
        
        batch, dim, num_depth, height, width = x.size()
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(batch*num_depth, dim, height, width) # [B*N,G,H,W]
        x =self.conv(x).view(batch, num_depth, height, width)
        '''
        # [B, N, H, W]
        x =self.conv(x).squeeze(1)
        '''
        # x = torch.sigmoid(x)
        x = torch.softmax(x,dim=1)
        # x = torch.clone(x)
        # x[x<0.1]=0.0
        # print("view weight, min:{:.2f},max{:.2f}".format(torch.min(x), torch.max(x)))
        # [B, H, W]
        x = torch.max(x, dim=1)[0]
        
        return x.unsqueeze(1)

class CorrNet(nn.Module):
    def __init__(self, G):
        super(CorrNet, self).__init__()
        # self.level = level
        self.conv0 = ConvReLU(G, 8)

        self.conv1 = ConvReLU(8, 16, stride=2)
        # self.conv2 = ConvReLU(16, 16)

        self.conv2 = ConvReLU(16, 32, stride=2)
        # self.conv4 = ConvReLU(32, 32)

        self.conv3 = nn.ConvTranspose2d(32, 16, 3, padding=1, output_padding=1,
                            stride=2, bias=False)

        self.conv4 = nn.ConvTranspose2d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False)

        self.conv5 = nn.Conv2d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        batch, dim, num_depth, height, width = x.size()
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(batch*num_depth, dim, height, width) # [B*N,G,H,W]
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        x = self.conv2(conv1)

        x = conv1 + self.conv3(x)
        del conv1
        x = conv0 + self.conv4(x)
        del conv0

        x = self.conv5(x).view(batch, num_depth, height, width)
        return x


class CorrNet3D(nn.Module):
    def __init__(self, G):
        super(CorrNet3D, self).__init__()
        # self.level = level
        self.conv0 = ConvReLU3D(G, 8)

        self.conv1 = ConvReLU3D(8, 16, stride=2)
        # self.conv2 = ConvReLU(16, 16)

        self.conv2 = ConvReLU3D(16, 32, stride=2)
        # self.conv4 = ConvReLU(32, 32)

        self.conv3 = nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                            stride=2, bias=False)

        self.conv4 = nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False)

        self.conv5 = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        batch, dim, num_depth, height, width = x.size()
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        x = self.conv2(conv1)

        x = conv1 + self.conv3(x)
        del conv1
        x = conv0 + self.conv4(x)
        del conv0

        x = self.conv5(x).view(batch, num_depth, height, width)
        return x



class RefineNet(nn.Module):
    def __init__(self, dim):
        super(RefineNet, self).__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, stride=1, padding=1, dilation=1),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(dim+16, 32, 3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, stride=1, padding=1, dilation=1),
            nn.Sigmoid()
        )

    def forward(self, depth, img_fea):
        
        depth_fea = self.depth_conv(depth)

        depth_refined = self.conv(torch.cat([depth_fea, img_fea], dim=1))
        
        return depth_refined
