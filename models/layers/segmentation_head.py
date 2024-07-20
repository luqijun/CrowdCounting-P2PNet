
import torch
import torch.nn as nn

class Segmentation_Head(nn.Module):

    def __init__(self, in_channels=[384, 192, 96], out_channels=1):
        super(Segmentation_Head, self).__init__()

        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels[0], in_channels[1], 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True),
        )
        # self.trans_conv1 = nn.Conv2d(in_channels[1] * 2, in_channels[1], kernel_size=1, stride=1, padding=0)
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels[1] * 2, in_channels[2], 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU(inplace=True),
        )

        # self.trans_conv2 = nn.Conv2d(in_channels[2] * 2, in_channels[2], kernel_size=1, stride=1, padding=0)
        # self.upsample3 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(in_channels[2] * 2, in_channels[2], 3, stride=1, padding=1),
        #     nn.BatchNorm2d(in_channels[2]),
        #     nn.ReLU(inplace=True),
        # )

        self.out_conv = nn.Conv2d(in_channels[2] * 2, out_channels, kernel_size=1, stride=1, padding=0)

        # nn.ReLU(inplace=True)

    def forward(self, input1, input2, input3):

        input3_up = self.upsample1(input3)
        input2 = torch.cat([input2, input3_up], dim=1)
        # input2 = self.trans_conv1(input2)

        input2_up = self.upsample2(input2)
        input1 = torch.cat([input1, input2_up], dim=1)
        # input1 = self.trans_conv2(input1)

        # input_1_up = self.upsample3(input1)
        output = self.out_conv(input1)
        return output