import torch
import torch.nn as nn

class resblock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, return_before_act=False, c_in_out=None, rep=False, branch_num=4):
        super(resblock, self).__init__()
        self.return_before_act = return_before_act
        if c_in_out is None:
            self.downsample = (in_channels != out_channels)
        else:
            pass
            # print('resblock: in1 = %d out1 = %d in2 = %d out2 = %d' % (
            # c_in_out[0][0], c_in_out[0][1], c_in_out[1][0], c_in_out[1][1]))
            self.downsample = (c_in_out[0][0] != c_in_out[1][1])

        if rep:
            pass
            # conv_module = Mutil_Branch_Conv2d
        else:
            conv_module = nn.Conv2d

        if self.downsample:
            if c_in_out is None:
                if rep:
                    self.conv1 = conv_module(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
                    self.ds = nn.Sequential(*[
                        conv_module(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                        nn.BatchNorm2d(out_channels)
                    ])
                else:
                    self.conv1 = conv_module(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
                    self.ds = nn.Sequential(*[
                        conv_module(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                        nn.BatchNorm2d(out_channels)
                    ])
            else:
                self.conv1 = conv_module(c_in_out[0][0], c_in_out[0][1], kernel_size=3, stride=2, padding=1, bias=False)
                self.ds = nn.Sequential(*[
                    conv_module(c_in_out[0][0], c_in_out[1][1], kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(c_in_out[1][1])
                ])
        else:
            if c_in_out is None:
                self.conv1 = conv_module(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            else:
                self.conv1 = conv_module(c_in_out[0][0], c_in_out[0][1], kernel_size=3, stride=1, padding=1, bias=False)
            self.ds = None
        if c_in_out is None:
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(c_in_out[0][1])

        self.relu = nn.ReLU()

        if c_in_out is None:
            self.conv2 = conv_module(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv2 = conv_module(c_in_out[1][0], c_in_out[1][1], kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(c_in_out[1][1])
        

    def forward(self, x):
        residual = x

        pout = self.conv1(x)  # pout: pre out before activation
        pout = self.bn1(pout)
        pout = self.relu(pout)

        pout = self.conv2(pout)
        pout = self.bn2(pout)

        if self.downsample:
            residual = self.ds(x)

        pout += residual

        out = self.relu(pout)

        if not self.return_before_act:
            return out
        else:
            return pout, out



class resnet20(nn.Module):
    def __init__(self, num_class, rep=False, branch_num=4):

        self.num_class = num_class

        # rep参数用于判断是否使用重参数机制
        super(resnet20, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.res1 = self.make_layer(resblock, 3, 16, 16, rep=rep, branch_num=branch_num)
        self.res2 = self.make_layer(resblock, 3, 16, 32, rep=rep, branch_num=branch_num)
        self.res3 = self.make_layer(resblock, 3, 32, 64, rep=rep, branch_num=branch_num)

        # self.avgpool = nn.AvgPool2d(8)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def make_layer(self, block, num, in_channels, out_channels, rep=False, branch_num=4):  # num must >=2
        layers = [block(in_channels, out_channels, False, rep=rep, branch_num=branch_num)]
        for i in range(num - 2):
            layers.append(block(out_channels, out_channels, False, rep=rep, branch_num=branch_num))
        layers.append(block(out_channels, out_channels, True, rep=rep, branch_num=branch_num))
        return nn.Sequential(*layers)

    def forward(self, x, output_only=True):
        pstem = self.conv1(x)  # pstem: pre stem before activation
        pstem = self.bn1(pstem)
        stem = self.relu(pstem)
        stem = (pstem, stem)

        rb1 = self.res1(stem[1])
        rb2 = self.res2(rb1[1])
        rb3 = self.res3(rb2[1])

        feat = self.avgpool(rb3[1])
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)

        if output_only:
            return feat, out
        return stem, rb1, rb2, rb3, feat, out

    def get_channel_num(self):
        return [16, 16, 32, 64, 64, self.num_class]

    def get_chw_num(self):
        return [(16, 32, 32),
                (16, 32, 32),
                (32, 16, 16),
                (64, 8, 8),
                (64,),
                (self.num_class,)]