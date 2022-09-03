import torch.nn as nn
import torch
import torch.nn.functional as F
# from model.networks.dropblock import DropBlock
from torch.distributions import Bernoulli


from loss import MarginCosineProduct

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1)))
            if torch.cuda.is_available():
                mask = mask.cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size), #- left_padding
            ]
        ).t()
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).long(), offsets.long()), 1)
        if torch.cuda.is_available():
            offsets = offsets.cuda()

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            #block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask#[:height, :width]
        return block_mask



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1, c_in_out = None):
        super(BasicBlock, self).__init__()

        if c_in_out is None:
            self.conv1 = conv3x3(inplanes, planes)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.LeakyReLU(0.1)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = conv3x3(planes, planes)
            self.bn3 = nn.BatchNorm2d(planes)
            self.maxpool = nn.MaxPool2d(stride)
            self.downsample = downsample
            self.stride = stride
            self.drop_rate = drop_rate
            self.num_batches_tracked = 0
            self.drop_block = drop_block
            self.block_size = block_size
            self.DropBlock = DropBlock(block_size=self.block_size)
        else:
            self.conv1 = conv3x3(c_in_out[0][0], c_in_out[0][1])
            self.bn1 = nn.BatchNorm2d(c_in_out[0][1])
            self.relu = nn.LeakyReLU(0.1)
            self.conv2 = conv3x3(c_in_out[1][0], c_in_out[1][1])
            self.bn2 = nn.BatchNorm2d(c_in_out[1][1])
            self.conv3 = conv3x3(c_in_out[2][0], c_in_out[2][1])
            self.bn3 = nn.BatchNorm2d(c_in_out[2][1])
            self.maxpool = nn.MaxPool2d(stride)
            self.downsample = downsample
            self.stride = stride
            self.drop_rate = drop_rate
            self.num_batches_tracked = 0
            self.drop_block = drop_block
            self.block_size = block_size
            self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class resnet12(nn.Module):

    def __init__(self, num_classes, block=BasicBlock, keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=5, cos_fc=True, cos_scale=32.0, cos_margin=0.1):
        super(resnet12, self).__init__()

        self.inplanes = 128
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        
        self.layer1 = self._make_layer(block, 128, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 256, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 512, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 1024, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)


        # self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        # self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        # self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        # self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        # self.feature_fc = nn.Linear(640, 1024)
        if avg_pool:
            # self.avgpool = nn.AvgPool2d(5, stride=1)
            self.avgpool = nn.AvgPool2d(12, stride=1)
        
        # self.fc = nn.Linear(1024, num_class)

        # 让resnet也采用教师网络的分类头结构
        self.cos_fc = cos_fc
        fc_bias = False
        self.head = nn.Linear(1024, num_classes, bias=fc_bias) if num_classes > 0 else nn.Identity()
        if self.cos_fc:
            self.head = MarginCosineProduct(1024, num_classes, scale=cos_scale, m=cos_margin)

        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, label):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.cos_fc:
            logits = self.head(x, label)
        else:
            logits = self.head(x)

        return x, logits


# def Res12(keep_prob=1.0, avg_pool=False, **kwargs):
#     """Constructs a ResNet-12 model.
#     """
#     model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
#     return model


# # 剪枝后的resnet12，结合输入的通道参数c_in_out定义
# class pruned_Resnet(nn.Module):

#     def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=5, c_in_out=None):
#         self.inplanes = 3
#         super(pruned_Resnet, self).__init__()

#         self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate, c_in_out = c_in_out[0:3])
#         self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate, c_in_out = c_in_out[3:6])
#         self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size, c_in_out = c_in_out[6:9])
#         self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size, c_in_out = c_in_out[9:12])
#         if avg_pool:
#             self.avgpool = nn.AvgPool2d(5, stride=1)
#         self.keep_prob = keep_prob
#         self.keep_avg_pool = avg_pool
#         self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
#         self.drop_rate = drop_rate

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1, c_in_out = None):

#         downsample = nn.Sequential(
#             nn.Conv2d(c_in_out[0][0], c_in_out[2][1], kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(c_in_out[2][1]),
#         )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, c_in_out))
        
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         # if self.keep_avg_pool:
#         #     x = self.avgpool(x)
#         # x = x.view(x.size(0), -1)
#         return x