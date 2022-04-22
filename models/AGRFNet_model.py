import torch
import torch.nn as nn
import torchvision.models as models
from .ResNet import ResNet50
from torch.nn import functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes//2, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention_cbam(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_cbam, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention_cbam(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_cbam, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Conv2dGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(Conv2dGRUCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.in_conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                                 out_channels=self.hidden_channels,
                                 kernel_size=self.kernel_size,
                                 stride=1,
                                 dilation=1,
                                 padding=self.padding,
                                 bias=self.bias)

        self.out_conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                                  out_channels=self.hidden_channels,
                                  kernel_size=self.kernel_size,
                                  stride=1,
                                  dilation=1,
                                  padding=self.padding,
                                  bias=self.bias)

        self.ca1 = ChannelAttention(input_channels*2)
        self.sa1 = SpatialAttention()

    def forward(self, input_tensor, hidden_state):
        h_cur = hidden_state
        combined = torch.cat((input_tensor, h_cur), dim=1)
        r = self.ca1(combined)
        z = self.sa1(combined)


        h_cur_bar = h_cur * r
        cc_h = self.out_conv(torch.cat((input_tensor, h_cur_bar), dim=1))
        h_bar = torch.tanh(cc_h)
        h_next = z * h_cur + (1 - z) * h_bar
        return h_next


class PixelAttention(nn.Module):
    def __init__(self, feat_in=64):
        super(PixelAttention, self).__init__()
        self.feat_in = feat_in

        self.h_c_convlayer = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.h_f_convlayer = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.spatial_pool_agr = nn.Sequential(
            nn.Conv2d(1936, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3),
            nn.Sigmoid(),
        )

        self.channel_pool_ag = nn.Sequential(
            nn.Linear(feat_in, feat_in // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_in // 4, feat_in),
        )

        self.channel_maxpool = nn.MaxPool2d(1)
        self.channel_avgpool = nn.AvgPool2d(1)
        self.channel_activation = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()


    def forward(self, depth, rgb):
        h_c = torch.cat((rgb, depth), 1)
        h_c_conv = self.h_c_convlayer(h_c)

        b, c, h, w = depth.shape
        kernel = depth.reshape(b, c, h * w).permute(0, 2, 1).reshape(-1, c, 1, 1)
        b, c, h, w = rgb.shape
        rgb_reshape = rgb.reshape(1, -1, h, w)
        pixel_corr = F.conv2d(rgb_reshape, kernel, groups=b).reshape(b, -1, h, w)

        b, c, h, w = pixel_corr.shape
        spatial_att_r = self.spatial_pool_agr(pixel_corr)

        b, c, h, w = depth.shape
        kernel = depth.reshape(b * c, 1, h, w)
        b, c, h, w = rgb.shape
        rgb_reshape = rgb.reshape(1, b * c, h, w)
        depth_corr = F.conv2d(rgb_reshape, kernel, groups=b * c)
        depth_corr = depth_corr.reshape(b, c, depth_corr.shape[-2], depth_corr.shape[-1])
        channel_max_pool = self.channel_maxpool(depth_corr).squeeze()
        channel_avg_pool = self.channel_avgpool(depth_corr).squeeze()
        channel_att = self.channel_activation(
            self.channel_pool_ag(channel_max_pool) + self.channel_pool_ag(channel_avg_pool)).unsqueeze(-1).unsqueeze(-1)

        h_c_conv_channel_att = h_c_conv * channel_att
        h_c_conv_spatial_att = h_c_conv * spatial_att_r

        h_attention_enhance_c = torch.cat((h_c_conv_spatial_att, h_c_conv_channel_att), 1)
        h_f_conv = self.h_f_convlayer(h_attention_enhance_c)
        return h_f_conv


class AGRFNet(nn.Module):
    def __init__(self, channel=32):
        super(AGRFNet, self).__init__()

        self.resnet = ResNet50('rgb')
        self.resnet_depth = ResNet50('rgbd')

        self.PixelAttention = PixelAttention()

        self.gru_1 = Conv2dGRUCell(input_channels=64, hidden_channels=64, kernel_size=3, bias=True)
        self.gru_2 = Conv2dGRUCell(input_channels=64, hidden_channels=64, kernel_size=3, bias=True)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.atten_depth_channel_0 = ChannelAttention_cbam(64)
        self.atten_depth_channel_1 = ChannelAttention_cbam(256)
        self.atten_depth_channel_2 = ChannelAttention_cbam(512)
        self.atten_depth_channel_3_1 = ChannelAttention_cbam(1024)
        self.atten_depth_channel_4_1 = ChannelAttention_cbam(2048)
        self.atten_depth_spatial_0 = SpatialAttention_cbam()
        self.atten_depth_spatial_1 = SpatialAttention_cbam()
        self.atten_depth_spatial_2 = SpatialAttention_cbam()
        self.atten_depth_spatial_3_1 = SpatialAttention_cbam()
        self.atten_depth_spatial_4_1 = SpatialAttention_cbam()

        self.atten_rgb_channel_0 = ChannelAttention_cbam(64)
        self.atten_rgb_channel_1 = ChannelAttention_cbam(256)
        self.atten_rgb_channel_2 = ChannelAttention_cbam(512)
        self.atten_rgb_channel_3_1 = ChannelAttention_cbam(1024)
        self.atten_rgb_channel_4_1 = ChannelAttention_cbam(2048)
        self.atten_rgb_spatial_0 = SpatialAttention_cbam()
        self.atten_rgb_spatial_1 = SpatialAttention_cbam()
        self.atten_rgb_spatial_2 = SpatialAttention_cbam()
        self.atten_rgb_spatial_3_1 = SpatialAttention_cbam()
        self.atten_rgb_spatial_4_1 = SpatialAttention_cbam()

        self.T_rgb_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.T_rgb_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.T_rgb_layer2_high = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.T_rgb_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2
        )
        self.T_rgb_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample4
        )

        self.T_depth_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.T_depth_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.T_depth_layer2_high = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.T_depth_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2
        )

        self.T_depth_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample4
        )

        self.deconv_layer_fuse = nn.Sequential(
            nn.Conv2d(in_channels=704, out_channels=384, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=384, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.predict_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True)
        )
        self.highlevel_predict_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True)
        )
        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth = self.resnet_depth.maxpool(x_depth)

        x1 = self.resnet.layer1(x)
        x1_depth = self.resnet_depth.layer1(x_depth)

        x2 = self.resnet.layer2(x1)
        x2_depth = self.resnet_depth.layer2(x1_depth)

        x3_1 = self.resnet.layer3_1(x2)
        x3_1_depth = self.resnet_depth.layer3_1(x2_depth)

        x4_1 = self.resnet.layer4_1(x3_1)
        x4_1_depth = self.resnet_depth.layer4_1(x3_1_depth)

        x_rgb_0 = x.mul(self.atten_rgb_channel_0(x))
        x_rgb_0 = x_rgb_0.mul(self.atten_rgb_spatial_0(x_rgb_0))
        x_rgb_0 = self.T_rgb_layer0(x_rgb_0)

        x_depth_0 = x_depth.mul(self.atten_depth_channel_0(x_depth))
        x_depth_0 = x_depth_0.mul(self.atten_depth_spatial_0(x_depth_0))
        x_depth_0 = self.T_depth_layer0(x_depth_0)

        x_rgb_1 = x1.mul(self.atten_rgb_channel_1(x1))
        x_rgb_1 = x_rgb_1.mul(self.atten_rgb_spatial_1(x_rgb_1))
        x_rgb_1 = self.T_rgb_layer1(x_rgb_1)
        x_depth_1 = x1_depth.mul(self.atten_depth_channel_1(x1_depth))
        x_depth_1 = x_depth_1.mul(self.atten_depth_spatial_1(x_depth_1))
        x_depth_1 = self.T_depth_layer1(x_depth_1)

        x_rgb_2 = x2.mul(self.atten_rgb_channel_2(x2))
        x_rgb_2 = x_rgb_2.mul(self.atten_rgb_spatial_2(x_rgb_2))
        x_rgb_2_1 = self.T_rgb_layer2_high(x_rgb_2)
        x_rgb_2_0 = self.upsample2(x_rgb_2_1)

        x_depth_2 = x2_depth.mul(self.atten_depth_channel_2(x2_depth))
        x_depth_2 = x_depth_2.mul(self.atten_depth_spatial_2(x_depth_2))
        x_depth_2_1 = self.T_depth_layer2_high(x_depth_2)
        x_depth_2_0 = self.upsample2(x_depth_2_1)

        x_rgb_3_1 = x3_1.mul(self.atten_rgb_channel_3_1(x3_1))
        x_rgb_3_1 = x_rgb_3_1.mul(self.atten_rgb_spatial_3_1(x_rgb_3_1))
        x_rgb_3_1 = self.T_rgb_layer3(x_rgb_3_1)
        x_depth_3_1 = x3_1_depth.mul(self.atten_depth_channel_3_1(x3_1_depth))
        x_depth_3_1 = x_depth_3_1.mul(self.atten_depth_spatial_3_1(x_depth_3_1))
        x_depth_3_1 = self.T_depth_layer3(x_depth_3_1)

        x_rgb_4_1 = x4_1.mul(self.atten_rgb_channel_4_1(x4_1))
        x_rgb_4_1 = x_rgb_4_1.mul(self.atten_rgb_spatial_4_1(x_rgb_4_1))
        x_rgb_4_1 = self.T_rgb_layer4(x_rgb_4_1)
        x_depth_4_1 = x4_1_depth.mul(self.atten_depth_channel_4_1(x4_1_depth))
        x_depth_4_1 = x_depth_4_1.mul(self.atten_depth_spatial_4_1(x_depth_4_1))
        x_depth_4_1 = self.T_depth_layer4(x_depth_4_1)

        h_s = self.PixelAttention(x_depth_4_1, x_rgb_4_1)
        h4_1 = self.gru_1(input_tensor=x_depth_4_1, hidden_state=h_s)
        h4 = self.gru_1(input_tensor=x_rgb_4_1, hidden_state=h4_1)

        h3_1 = self.gru_1(input_tensor=x_depth_3_1, hidden_state=h4)
        h3 = self.gru_1(input_tensor=x_rgb_3_1, hidden_state=h3_1)

        h2_1 = self.gru_1(input_tensor=x_depth_2_1, hidden_state=h3)
        h2_1 = self.gru_1(input_tensor=x_rgb_2_1, hidden_state=h2_1)

        h2_1 = self.upsample2(h2_1)

        h = self.highlevel_predict_layer(h2_1)
        attention = h.sigmoid()

        x_rgb_2_0 = attention.mul(x_rgb_2_0) + x_rgb_2_0
        x_depth_2_0 = attention.mul(x_depth_2_0) + x_depth_2_0
        x_rgb_1 = attention.mul(x_rgb_1) + x_rgb_1
        x_depth_1 = attention.mul(x_depth_1) + x_depth_1
        x_rgb_0 = attention.mul(x_rgb_0) + x_rgb_0
        x_depth_0 = attention.mul(x_depth_0) + x_depth_0

        h2_0 = self.gru_2(input_tensor=x_depth_2_0, hidden_state=h2_1)
        h2 = self.gru_2(input_tensor=x_rgb_2_0, hidden_state=h2_0)

        h1_0 = self.gru_2(input_tensor=x_depth_1, hidden_state=h2)
        h1 = self.gru_2(input_tensor=x_rgb_1, hidden_state=h1_0)

        h0_0 = self.gru_2(input_tensor=x_depth_0, hidden_state=h1)
        h0 = self.gru_2(input_tensor=x_rgb_0, hidden_state=h0_0)

        x_rgb_3_1 = self.upsample2(x_rgb_3_1)
        x_depth_3_1 = self.upsample2(x_depth_3_1)
        x_rgb_4_1 = self.upsample2(x_rgb_4_1)
        x_depth_4_1 = self.upsample2(x_depth_4_1)
        ca_r = torch.cat((x_rgb_0, x_rgb_1, x_rgb_2_0, x_rgb_3_1, x_rgb_4_1), 1)
        ca_d = torch.cat((x_depth_0, x_depth_1, x_depth_2_0, x_depth_3_1, x_depth_4_1), 1)
        z = torch.cat((h0, ca_r, ca_d), 1)

        y = self.deconv_layer_fuse(z)
        y = y + self.upsample2(h0)
        y = self.predict_layer(y)

        return y, self.upsample4(h)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            if k == 'conv1.weight':
                all_params[k] = torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)


