from torchvision import models
import torch.nn.functional as F
from .resnet_model import *


class ERM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ERM, self).__init__()
        self.adaptEdge = nn.Conv2d(1, in_channel, 3, padding=1)
        self.avg = nn.AvgPool2d((3, 3), stride=1,padding=1)
        self.getEdge = nn.ReLU(inplace=True)
        self.getFinalEdge = nn.Sequential(nn.Conv2d(2 * in_channel, out_channel, 3, padding=1),
                                          nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.getFinalInfo = nn.Sequential(nn.Conv2d(3 * in_channel, out_channel, 3, padding=1),
                                          nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

    def forward(self, rgb_deep, rgb_shallow, edge):
        edge_rgb = self.getEdge(torch.sub(rgb_shallow, self.avg(rgb_shallow)))
        edge_final = self.getFinalEdge(torch.cat((self.adaptEdge(edge), edge_rgb), 1))
        final = self.getFinalInfo(torch.cat((rgb_deep, rgb_shallow, edge_final), 1))

        return final

    def initialize(self):
        weight_init(self)


class SubNet(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(SubNet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        self.down2 = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.down3 = nn.Upsample(scale_factor=0.25, mode='bilinear')
        self.down4 = nn.Upsample(scale_factor=0.125, mode='bilinear')

        #####
        self.Decoder4 = ERM(64, 64)
        self.Decoder3 = ERM(64, 64)
        self.Decoder2 = ERM(64, 64)
        self.Decoder1 = ERM(64, 64)

        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.d1conv = nn.Conv2d(64, 1, 3, padding=1)
        self.d2conv = nn.Conv2d(64, 1, 3, padding=1)
        self.d3conv = nn.Conv2d(64, 1, 3, padding=1)
        self.d4conv = nn.Conv2d(64, 1, 3, padding=1)
        self.up4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, edge):

        hx = x
        h1_edge = edge
        h2_edge = self.down2(edge)
        h3_edge = self.down3(edge)
        h4_edge = self.down4(edge)

        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.Decoder4(hx, hx4, h4_edge)
        hx = self.upscore2(d4)

        d3 = self.Decoder3(hx, hx3, h3_edge)
        hx = self.upscore2(d3)

        d2 = self.Decoder2(hx, hx2, h2_edge)
        hx = self.upscore2(d2)

        d1 = self.Decoder1(hx, hx1, h1_edge)

        residual = self.conv_d0(d1)

        return x + residual, self.d1conv(d1), self.up2(self.d2conv(d2)), self.up3(self.d3conv(d3)), self.up4(self.d4conv(d4))


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class MSF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSF, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.branch1 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1),
                                     nn.BatchNorm2d(out_channel), nn.ReLU(True))
        self.branch2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, padding=2, dilation=2),
                                     nn.BatchNorm2d(out_channel), nn.ReLU(True))
        self.branch3 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, padding=4, dilation=4),
                                     nn.BatchNorm2d(out_channel), nn.ReLU(True))
        self.branch4 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, padding=6, dilation=6),
                                     nn.BatchNorm2d(out_channel), nn.ReLU(True),)
        self.score = nn.Conv2d(out_channel*4, 1, 3, padding=1)

    def forward(self, x):
        x = self.relu(self.bn(self.convert(x)))
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.score(x)

        return x

    def initialize(self):
        weight_init(self)


class EFBI(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EFBI, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True)
        )
        self.channel = out_channel
        self.convs1 = nn.Sequential(nn.Conv2d(in_channel, 2 * out_channel, 3, padding=1), nn.BatchNorm2d(2 * out_channel), nn.ReLU(True))
        self.convs2 = nn.Sequential(nn.Conv2d(2 * out_channel, 2 * out_channel, 3, padding=1), nn.BatchNorm2d(2 * out_channel), nn.ReLU(True))

        self.convs3 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True))
        self.convs4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True)
        )
        self.convs5 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True)
        )

    def forward(self, x, y, edge):
        a = torch.sigmoid(-y)

        x = self.relu(self.bn(self.convert(x)))
        x = a.expand(-1, self.channel, -1, -1).mul(x)
        bg = self.convs(x)

        tmp = torch.sigmoid(-edge)
        tmp = tmp.expand(-1, self.channel, -1, -1).mul(self.convs3(y))
        fg = torch.add(self.convs4(tmp), edge)
        result = torch.add(bg, self.convs5(fg))

        return result

    def initialize(self):
        weight_init(self)


class RCU(nn.Module):
    def __init__(self, InChannel):
        super(RCU, self).__init__()
        self.Conv1 = nn.Sequential(nn.BatchNorm2d(InChannel), nn.ReLU(inplace=True), nn.Conv2d(InChannel, InChannel, 3, padding=1))
        self.Conv2 = nn.Sequential(nn.BatchNorm2d(InChannel), nn.ReLU(inplace=True), nn.Conv2d(InChannel, InChannel, 3, padding=1))

    def forward(self, x):
        res = x
        x = self.Conv2(self.Conv1(x))

        return x + res


class TSERNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(TSERNet, self).__init__()

        resnet = models.resnet34(pretrained=True)

        # -------------Encoder--------------

        self.inconv = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        # stage 1
        self.encoder1 = resnet.layer1  # 224 * 224 * 64
        # stage 2
        self.encoder2 = resnet.layer2  # 112 * 112 * 128
        # stage 3
        self.encoder3 = resnet.layer3  # 56 * 56 * 256
        # stage 4
        self.encoder4 = resnet.layer4  # 28 * 28 * 512

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 5
        self.resb5_1 = BasicBlock(512, 512)
        self.resb5_2 = BasicBlock(512, 512)
        self.resb5_3 = BasicBlock(512, 512)  # 14 * 14 * 512

        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 6
        self.resb6_1 = BasicBlock(512, 512)
        self.resb6_2 = BasicBlock(512, 512)
        self.resb6_3 = BasicBlock(512, 512)  # 7 * 7 * 512

        # -------------Fusion--------------
        self.Resume6_1 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512),
                                       nn.ReLU(inplace=True))
        self.Resume6_2 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512),
                                       nn.ReLU(inplace=True))
        self.Resume6_3 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512),
                                       nn.ReLU(inplace=True))
        self.getHigherSideInfo = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(inplace=True),
                                               nn.Upsample(scale_factor=16, mode='bilinear'))
        self.MSF = MSF(128, 64)
        self.adjust = nn.Conv2d(1, 128, 3, padding=1)
        self.Enout_edge = nn.Conv2d(128, 1, 3, padding=1)
        self.up_edge = nn.Upsample(scale_factor=2, mode='bilinear')

        self.RCU1 = RCU(64)
        self.RCU2 = RCU(128)
        self.RCU3 = RCU(256)
        self.RCU4 = RCU(512)
        self.RCU5 = RCU(512)
        self.RCU6 = RCU(512)

        # -------------Bridge--------------

        # stage Bridge
        self.convbg_1 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        # ------------RA Module-------------

        self.EFBI6 = EFBI(512, 512)
        self.EFBI5 = EFBI(512, 512)
        self.EFBI4 = EFBI(512, 512)
        self.EFBI3 = EFBI(256, 256)
        self.EFBI2 = EFBI(128, 128)
        self.EFBI1 = EFBI(64, 64)

        self.downEdge6 = nn.Upsample(scale_factor=0.03125, mode='bilinear')
        self.downEdge5 = nn.Upsample(scale_factor=0.0625, mode='bilinear')
        self.downEdge4 = nn.Upsample(scale_factor=0.125, mode='bilinear')
        self.downEdge3 = nn.Upsample(scale_factor=0.25, mode='bilinear')
        self.downEdge2 = nn.Upsample(scale_factor=0.5, mode='bilinear')

        # -------------Decoder--------------

        # stage 6d
        self.conv6d_1 = nn.Conv2d(512, 512, 3, padding=1)  # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  ###
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        # stage 5d
        self.conv5d_1 = nn.Conv2d(512, 512, 3, padding=1)  # 16
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        # stage 4d
        self.conv4d_1 = nn.Conv2d(512, 512, 3, padding=1)  # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        # stage 3d
        self.conv3d_1 = nn.Conv2d(256, 256, 3, padding=1)  # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256, 256, 3, padding=1)  ###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        # stage 2d

        self.conv2d_1 = nn.Conv2d(128, 128, 3, padding=1)  # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128, 128, 3, padding=1)  ###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        # stage 1d
        self.conv1d_1 = nn.Conv2d(64, 64, 3, padding=1)  # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64, 64, 3, padding=1)  ###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # -------------Side Output--------------
        self.outconvb = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv6 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv5 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, 1, 3, padding=1)

        # -------------Refine Module-------------

        self.subnet = SubNet(1, 64)

    def forward(self, x):
        hx = x

        # -------------Encoder-------------

        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        h1 = self.encoder1(hx)  # stage1
        h2 = self.encoder2(h1)  # stage2
        h3 = self.encoder3(h2)  # stage3
        h4 = self.encoder4(h3)

        hx = self.pool4(h4)  # stage4

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx = self.pool5(h5)  # stage5

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)  # stage6

        # -------------Fusion-------------

        h1 = self.RCU1(h1)
        h2 = self.RCU2(h2)
        h3 = self.RCU3(h3)
        h4 = self.RCU4(h4)
        h5 = self.RCU5(h5)
        h6 = self.RCU6(h6)

        c2 = h2
        c6 = h6
        f6 = self.Resume6_3(self.Resume6_2(self.Resume6_1(c6)))
        f_HigherSideInfo = self.getHigherSideInfo(f6)
        edge_sup = torch.add(c2, f_HigherSideInfo)
        edge_sup = self.adjust(self.MSF(edge_sup))
        edge_sup = self.up_edge(self.Enout_edge(edge_sup))

        edge_de = edge_sup
        edge_ref = edge_sup

        # -------------Bridge-------------

        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6)))
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))  # bridge

        # -------------Decoder----------
        h_6 = self.EFBI6(h6, hbg, self.downEdge6(edge_de))

        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(h_6)))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6)

        h_5 = self.EFBI5(h5, hx, self.downEdge5(edge_de))

        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(h_5)))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5)

        h_4 = self.EFBI4(h4, hx, self.downEdge4(edge_de))

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(h_4)))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4)

        h_3 = self.EFBI3(h3, hx, self.downEdge3(edge_de))

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(h_3)))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3)

        h_2 = self.EFBI2(h2, hx, self.downEdge2(edge_de))

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(h_2)))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2)

        h_1 = self.EFBI1(h1, hx, edge_de)

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(h_1)))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        # -------------Side Output-------------
        db = self.outconvb(hbg)
        db = self.upscore6(db)

        d6 = self.outconv6(hd6)
        d6 = self.upscore6(d6)

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)

        d1 = self.outconv1(hd1)

        # -------------Refine Module-------------

        dout, d1_ref, d2_ref, d3_ref, d4_ref = self.subnet(d1, edge_ref)

        return F.sigmoid(dout), \
               F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6), F.sigmoid(db), \
               F.sigmoid(edge_sup), \
               F.sigmoid(d1_ref), F.sigmoid(d2_ref), F.sigmoid(d3_ref), F.sigmoid(d4_ref)
