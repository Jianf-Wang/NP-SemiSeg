import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import ASPP, get_syncbn
from .np_head import *


class dec_deeplabv3(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
    ):
        super(dec_deeplabv3, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )
        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        aspp_out = self.aspp(x)
        res = self.head(aspp_out)
        return res


class dec_deeplabv3_plus(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
        rep_head=True,
    ):
        super(dec_deeplabv3_plus, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.rep_head = rep_head

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.classifier1 = NP_HEAD(input_dim=512, num_classes=num_classes)

        if self.rep_head:

            self.representation = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            )

    def forward(self, x, deterministic_memory, latent_memory, x_context_in=None, labels_target_in=None, labels_context_in=None, forward_times=5, phase_train=True):
        x1, x2, x3, x4 = x
        aspp_out = self.aspp(x4)
        low_feat = self.low_conv(x1)
        aspp_out = self.head(aspp_out)
        h, w = low_feat.size()[-2:]
        aspp_out = F.interpolate(
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )
        aspp_out = torch.cat((low_feat, aspp_out), dim=1)

       

        if x_context_in is not None:
           x1, x2, x3, x4 = x_context_in
           aspp_out1 = self.aspp(x4)
           low_feat1 = self.low_conv(x1)
           aspp_out1 = self.head(aspp_out1)
           h, w = low_feat1.size()[-2:]
           aspp_out1 = F.interpolate(
            aspp_out1, size=(h, w), mode="bilinear", align_corners=True
           )
           aspp_out1 = torch.cat((low_feat1, aspp_out1), dim=1)
           x_context_in = aspp_out1
        


        if phase_train:
           
           output, mean, sigma, mean_c, sigma_c, deterministic_memory, latent_memory  = self.classifier1(aspp_out, deterministic_memory, latent_memory,  x_context_in=x_context_in, labels_target_in=labels_target_in, labels_context_in=labels_context_in, forward_times=forward_times, phase_train=phase_train)
           res={'pred': output}
           res['mean_t'] = mean
           res['sigma_t'] = sigma
           res['mean_c'] = mean_c
           res['sigma_c'] = sigma_c
           res['deterministic_memory'] = deterministic_memory
           res['latent_memory'] = latent_memory
        else:
           output = self.classifier1(aspp_out, deterministic_memory, latent_memory, forward_times=forward_times, phase_train=phase_train)
           res={'pred': output}
        #res = {"pred": self.classifier1(aspp_out)}

        if self.rep_head:
           res["rep"] = self.representation(aspp_out)
        
        return res


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, sync_bn=False):
        super(Aux_Module, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.aux = nn.Sequential(
            nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        res = self.aux(x)
        return res
