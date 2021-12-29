#!/usr/bin/env python3
# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang
#                                                    Mingshuang Luo)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class VisualNet2(nn.Module):
    def __init__(self, num_classes):
        super(VisualNet2, self).__init__()
        self.num_classes = num_classes
        self.inputDim = 512
        self.conv3d = nn.Conv3d(
            3,
            64,
            kernel_size=(5, 7, 7),
            stride=(1, 2, 2),
            padding=(2, 3, 3),
            bias=False,
        )
        self.bn = nn.BatchNorm3d(64, track_running_stats=True)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        # resnet
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])

        # grus
        self.gru1 = nn.GRU(512, 512, 1, bidirectional=True)
        self.gru2 = nn.GRU(1024, 512, 1, bidirectional=True)

        # dropout
        self.dropout = nn.Dropout(p=0.5)

        # fc
        self.linear = nn.Linear(1024, self.num_classes)

        # initialize
        self._initialize_weights()

    def forward(self, x):
        frameLen = x.size(2)
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet18(x)

        x = self.dropout(x)
        x = x.view(-1, frameLen, self.inputDim)

        x = x.permute(1, 0, 2)
        x, h = self.gru1(x)
        x, h = self.gru2(self.dropout(x))
        x = self.linear(x)
        x = x.permute(1, 0, 2)
        x = nn.functional.log_softmax(x, dim=-1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
