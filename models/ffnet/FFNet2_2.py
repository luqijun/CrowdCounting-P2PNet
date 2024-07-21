from torchvision import models
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
from models.ffnet.ODConv2d import ODConv2d
from util.misc import NestedTensor

from ..backbone import build_backbone
from ..matcher import build_matcher_crowd
from ..loss import SetCriterion_Crowd
from ..layers import RegressionModel, ClassificationModel, AnchorPoints, Segmentation_Head

__all__ = ['FFNet']

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
 
    def forward(self, x):
        return torch.Tensor.permute(x, self.dims)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)      
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps) 
        x = x.permute(0, 3, 1, 2)      
        return x
    
def conv(in_ch, out_ch, ks, stride):
    
    pad = (ks - 1) // 2
    stage = nn.Sequential(nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ks, stride=stride,
                                       padding=pad, bias=False),
                          LayerNorm2d((out_ch,), eps=1e-06, elementwise_affine=True),
                          nn.GELU(approximate='none'))
    return stage

class ChannelAttention(nn.Module):  
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=self.avg_pool(x)
        avgout = self.shared_MLP(x)
        return self.sigmoid(avgout)
    
class SpatialAttention(nn.Module):  
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.conv1(x)
        return self.sigmoid(x)

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone,self).__init__()
        
        feats =list(convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features.children())
        
        self.stem = nn.Sequential(*feats[0:2])
        self.stage1 = nn.Sequential(*feats[2:4])
        self.stage2 = nn.Sequential(*feats[4:6])
        # self.stage3 = nn.Sequential(*feats[6:12])
        
    def forward(self, x):
        x = x.float()
        x = self.stem(x)
        feature0 = x
        x = self.stage1(x)
        feature1 = x
        x = self.stage2(x)
        feature2 = x
        # x = self.stage3(x)
        
        return feature0, feature1, feature2

class ccsm(nn.Module):
    def __init__(self, channel, channel2, num_filters):
        super(ccsm, self).__init__()
        self.ch_att_s = ChannelAttention(channel)
        self.sa_s = SpatialAttention(7)
        self.conv1 = nn.Sequential(
            ODConv2d(channel, channel, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = channel))
        self.conv2 = nn.Sequential(
            ODConv2d(channel, channel2, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = channel2))
        
        self.conv3 = nn.Sequential(
            ODConv2d(channel2, channel2, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = channel2))
        self.conv4 = nn.Sequential(
            ODConv2d(channel2, num_filters, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = num_filters))
           
    def forward(self, x):
        x = self.ch_att_s(x)*x
        pool1 = x
        x = self.conv1(x)
        x = x + pool1
        x = self.conv2(x)
        pool2 = x
        x = self.conv3(x)
        x = x + pool2
        x = self.conv4(x)
        
        x = self.sa_s(x)*x

        return x
    
class Fusion(nn.Module):
    def __init__(self, num_filters1, num_filters2, num_filters3, out_channels=1):
        super(Fusion, self).__init__()
        # self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample_1 = nn.ConvTranspose2d(in_channels=num_filters2, out_channels=num_filters2, kernel_size=4, padding=1, stride=2)
        self.upsample_2 = nn.ConvTranspose2d(in_channels=num_filters3, out_channels=num_filters3, kernel_size=4, padding=0, stride=4)
        # self.upsample_2 = nn.ConvTranspose2d(in_channels=num_filters3, out_channels=num_filters3, kernel_size=4, padding=1, stride=2)
        self.final = nn.Sequential(
            nn.Conv2d(num_filters1+num_filters2+num_filters3, out_channels, kernel_size=1, padding=0),
            nn.ReLU(),
        )
        
    def forward(self, x1, x2, x3):
        # x1 = self.down_sample(x1)
        x2 = self.upsample_1(x2)
        x3 = self.upsample_2(x3)
        # x3 = self.upsample_2(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.final(x)
        
        return x


# 使用1/4特征预测
class FFNet2_2(nn.Module):
    def __init__(self):
        super().__init__()
        out_channels = 16 + 32 + 64
        num_filters = [16, 32, 64]
        self.backbone = Backbone()

        self.seg_head = Segmentation_Head(in_channels=[384, 192, 96])

        self.ccsm1 = ccsm(96, 38, num_filters[0])
        self.ccsm2 = ccsm(192, 96, num_filters[1])
        self.ccsm3 = ccsm(384, 192, num_filters[2])
        self.fusion = Fusion(num_filters[0], num_filters[1], num_filters[2], out_channels=out_channels)

        row = 1
        line = 1
        self.num_classes = 2
        # the number of all anchor points
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=out_channels, num_anchor_points_list=num_anchor_points)
        self.classification = ClassificationModel(num_features_in=out_channels, \
                                                  num_classes=self.num_classes, \
                                                  num_anchor_points_list=num_anchor_points)

        self.anchor_points = AnchorPoints(pyramid_levels=[2, ], row=row, line=line)


    def forward(self, samples: NestedTensor):
        x = samples.tensors
        pool1, pool2, pool3 = self.backbone(x)

        pred_seg_map = self.seg_head(pool1, pool2, pool3)
        seg_attention = pred_seg_map.sigmoid()
        pool1 = pool1 * F.interpolate(seg_attention, size=pool1.shape[-2:])
        pool2 = pool2 * F.interpolate(seg_attention, size=pool2.shape[-2:])
        pool3 = pool3 * F.interpolate(seg_attention, size=pool3.shape[-2:])

        pool1 = self.ccsm1(pool1)
        pool2 = self.ccsm2(pool2)
        pool3 = self.ccsm3(pool3)
        features = self.fusion(pool1, pool2, pool3)

        batch_size = x.shape[0]
        # run the regression and classification branch
        regression = self.regression(features) * 100  # 8x
        classification = self.classification(features)
        anchor_points = self.anchor_points(x).repeat(batch_size, 1, 1)
        # decode the points as prediction
        output_coord = regression + anchor_points
        output_class = classification
        out = {
            'pred_logits': output_class,
            'pred_points': output_coord,
            'anchor_points': anchor_points,
            'pred_seg_map': pred_seg_map
        }

        return out
 
        # B, C, H, W = x.size()
        # x_sum = x.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # x_normed = x / (x_sum + 1e-6)
        #
        # return x, x_normed


def build_ffnet2_2(args, training):

    # treats persons as a single class
    num_classes = 1

    model = FFNet2_2()
    if not training:
        return model

    weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef, 'loss_seg_head': 0.1}
    losses = ['labels', 'points', 'seg_head']
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(num_classes, \
                                matcher=matcher, weight_dict=weight_dict, \
                                eos_coef=args.eos_coef, losses=losses)

    return model, criterion



if __name__ == '__main__':
    x = torch.rand(size=(16, 3, 512, 512), dtype=torch.float32)
    model = FFNet2_2()
    
    mu, mu_norm = model(x)
    print(mu.size(), mu_norm.size())
    