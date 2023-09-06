
from functools import partial
import torch.nn.functional as F
import torch
from einops import rearrange, repeat, reduce
from torch import nn,Tensor
from torchvision.transforms import ToTensor
from torch.nn.functional import sigmoid, relu, prelu
import time
from torchvision.models import resnet50, ResNet50_Weights
from torch.autograd import Variable
from typing import Optional


class FeatureFusionModule(nn.Module):

    def __init__(self, input_modal, transform_dim, kernel_size=(7, 7), use_attention=False, **kwargs):

        super(FeatureFusionModule, self).__init__()

        assert type(input_modal) == list
        assert transform_dim in [64, 128, 256]
        self.input_modal = input_modal
        self.input_modal_count = len(self.input_modal)
        self.transform_dim = transform_dim

        self.feature_transforms = nn.ModuleDict()
        for modal in self.input_modal:
            if modal == 'saliency':
                self.feature_transforms[modal] = self.make_merge_layer(1, transform_dim, 7, 2, 3)
            else:
                self.feature_transforms[modal] = self.make_merge_layer(3, transform_dim, 7, 2, 3)
        self.use_attention = use_attention
        self.concat_dim = self.input_modal_count * transform_dim
        if use_attention:
            self.channal_attention = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Conv3d(in_channels=self.concat_dim,
                          out_channels=self.concat_dim // 4,
                          kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1)),
                nn.Sigmoid(),
                nn.Conv3d(in_channels=self.concat_dim // 4,
                          out_channels=self.concat_dim,
                          kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1)),
                nn.Sigmoid()
            )


    def make_merge_layer(self, in_channel, hidden_dim, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=hidden_dim, kernel_size=(1, kernel_size, kernel_size),
                      stride=(1, stride, stride), padding=(0, padding, padding)),
            nn.BatchNorm3d(hidden_dim)
        )

    def forward(self, frame, flow, seg, saliency):
        key_dict = {'frame': frame, 'flow': flow, 'seg': seg, 'saliency': saliency}
        transform_result = torch.concat([self.feature_transforms[modal](key_dict[modal]) for modal in self.input_modal],
                                        dim=1)

        if self.use_attention:
            attention = self.channal_attention(transform_result)
            return transform_result, attention
        else:
            return transform_result,None

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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


class Bottleneck_R3D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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

        return out

class R3D_Backbone(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,**kwargs
                 ):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.layer_num=kwargs['layer_num']
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        if self.layer_num >= 2:
            self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                           shortcut_type)
        if self.layer_num>=3:
            self.layer2 = self._make_layer(block,
                                           block_inplanes[1],
                                           layers[1],
                                           shortcut_type,
                                           stride=2)
        if self.layer_num >= 4:
            self.layer3 = self._make_layer(block,
                                           block_inplanes[2],
                                           layers[2],
                                           shortcut_type,
                                           stride=2)
        if self.layer_num >= 5:
            self.layer4 = self._make_layer(block,
                                           block_inplanes[3],
                                           layers[3],
                                           shortcut_type,
                                           stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        if self.layer_num==2:
            return x
        x = self.layer2(x)

        if self.layer_num == 3:
            return x
        x = self.layer3(x)

        if self.layer_num == 4:
            return x

        x = self.layer4(x)
        return x

class C3D_Backbone(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(C3D_Backbone, self).__init__()
        self.input_dim = input_dim
        self.relu = nn.ReLU()
        self.layer_num = kwargs['layer_num']
        assert self.layer_num in [2, 3, 4, 5]
        self.conv1 = nn.Conv3d(self.input_dim, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        if self.layer_num >= 2:
            self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        if self.layer_num >= 3:
            self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        if self.layer_num >= 4:
            self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        if self.layer_num >= 5:
            self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        if self.layer_num == 2:
            return x
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        if self.layer_num == 3:
            return x
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        if self.layer_num == 4:
            return x
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        return x

class Res_Pyramid(nn.Module):
    def __init__(self, resnet50_mode='pretrain', **kwargs):
        super(Res_Pyramid, self).__init__()
        assert resnet50_mode in ['random', 'pretrain', 'frozen']
        self.resnet50_mode = resnet50_mode
        if resnet50_mode == 'pretrain' or resnet50_mode == 'frozen':
            print('load resnet50 pretrain')
            self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.resnet = resnet50(weights=None)
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]

    def forward(self, data):
        if self.resnet50_mode == 'frozen':
            with torch.no_grad():
                feat_map = self.conv1(data)
                feat_map2 = self.conv2(feat_map)
                feat_map3 = self.conv3(feat_map2)
                feat_map4 = self.conv4(feat_map3)
        else:
            feat_map = self.conv1(data)
            feat_map2 = self.conv2(feat_map)
            feat_map3 = self.conv3(feat_map2)
            feat_map4 = self.conv4(feat_map3)
        return feat_map2, feat_map3, feat_map4


class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )

    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h

    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h

    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)

        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)


class BottleneckBlock_Unet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gru = ConvGRU(channels // 2)

    def forward(self, x, r: Optional[Tensor] = None):
        a, b = x.split(self.channels // 2, dim=2)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=2)

        return x, r


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, src_channels):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels + src_channels, src_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(src_channels),
            nn.ReLU(True),
            nn.Conv3d(src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )

    def forward_time_series(self, x, f):
        B, T, _, H, W = x.shape
        x = torch.concat([x, f], dim=2)
        x = rearrange(x, 'b t c h w->b c t h w')
        x = self.conv(x)
        return x

    def forward(self, x, s):
        return self.forward_time_series(x, s)



class ClassifierModule(nn.Module):
    def __init__(self, in_channels, attention=False, hidden_dim=16, **kwargs):
        super(ClassifierModule, self).__init__()
        self.class_num = kwargs['class_num']
        self.in_channels = in_channels
        self.attention = attention
        self.conv = nn.Sequential(
            nn.Conv3d(self.in_channels, hidden_dim, kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(hidden_dim, 16, kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        if attention:
            self.attention_module = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Conv3d(in_channels=self.in_channels,
                          out_channels=self.in_channels // 4,
                          kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=self.in_channels // 4,
                          out_channels=self.in_channels,
                          kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1)),
                nn.Sigmoid()
            )
            pass
        self.fc = nn.Linear(16, self.class_num)

    def forward(self, feature):
        B, C, T, H, W = feature.shape
        if self.attention:
            feature = self.attention_module(feature) * feature
        result = self.conv(feature)
        result = result.squeeze()
        return self.fc(result)


class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)

    def forward_single_frame(self, s0):
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        return s1, s2

    def forward_time_series(self, s0):
        B, T = s0.shape[:2]
        s0 = s0.flatten(0, 1)
        s1, s2 = self.forward_single_frame(s0)
        s1 = s1.unflatten(0, (B, T))
        s2 = s2.unflatten(0, (B, T))
        return s1, s2

    def forward(self, s0):
        if s0.ndim == 5:
            return self.forward_time_series(s0)
        else:
            return self.forward_single_frame(s0)


class LightFramework(nn.Module):
    def __init__(self, **kwargs):
        super(LightFramework, self).__init__()
        self.params = kwargs
        assert 'input_modal' in self.params
        self.modal = self.params['input_modal']
        self.use_attention = self.params['use_attention']
        self.use_res = self.params['use_res']
        self.use_pretrain = self.params['use_pretrain']
        self.layer_num = self.params['layer_num']
        self.backbone = self.params['backbone']
        self.modal_count=len(self.modal)
        self.concat_dim = self.modal_count * self.params['transform_dim']
        self.clip_n = self.params['clip_n']
        self.transform_dim = self.params['transform_dim']

        assert self.backbone in ['r3d', 'c3d','unet']
        self.modal_count = len(self.modal)
        self.merge_module = FeatureFusionModule(**kwargs)

        if self.backbone == 'r3d':
            model_configs = {
                'n_input_channels': self.concat_dim,
                'shortcut_type': 'B',
                'conv1_t_size': 7,
                'conv1_t_stride': 1,
                'no_max_pool': False,
                'widen_factor': 1.0,
            }
            self.feature_extract = R3D_Backbone(Bottleneck_R3D, [3, 4, 6, 3], [64, 128, 256, 512], **model_configs, **kwargs)
            if self.use_pretrain:
                model_path = r'r3d50_KMS_200ep.pth'
                p_dict = torch.load(model_path)
                m_dict = self.state_dict()
                for m in m_dict:
                    if 'feature_extract.' not in m:
                        continue
                    m_name=m.replace('feature_extract.','')
                    if m_name in p_dict['state_dict'] and m_dict[m].shape==p_dict['state_dict'][m_name].shape:
                        print('load weight:',m_name)
                        m_dict[m]=p_dict['state_dict'][m_name]

            self.layer_shape = {
                2: (256, self.clip_n // 2, 14, 24),
                3: (512, self.clip_n // 4, 7, 12),
                4: (1024, self.clip_n // 8, 4, 6),
                5: (2048, self.clip_n // 16 if self.clip_n==16 else self.clip_n // 8, 2, 3),
            }
        if self.backbone == 'c3d':
            self.feature_extract = C3D_Backbone(input_dim=self.concat_dim, **kwargs)
            if self.use_pretrain:
                model_path = r'c3d-pretrained.pth'
                corresp_name = {
                    "features.0.weight": "feature_extract.conv1.weight",
                    "features.0.bias": "feature_extract.conv1.bias",
                    "features.3.weight": "feature_extract.conv2.weight",
                    "features.3.bias": "feature_extract.conv2.bias",
                    "features.6.weight": "feature_extract.conv3a.weight",
                    "features.6.bias": "feature_extract.conv3a.bias",
                    "features.8.weight": "feature_extract.conv3b.weight",
                    "features.8.bias": "feature_extract.conv3b.bias",
                    "features.11.weight": "feature_extract.conv4a.weight",
                    "features.11.bias": "feature_extract.conv4a.bias",
                    "features.13.weight": "feature_extract.conv4b.weight",
                    "features.13.bias": "feature_extract.conv4b.bias",
                    "features.16.weight": "feature_extract.conv5a.weight",
                    "features.16.bias": "feature_extract.conv5a.bias",
                    "features.18.weight": "feature_extract.conv5b.weight",
                    "features.18.bias": "feature_extract.conv5b.bias",
                    "classifier.0.weight": "feature_extract.fc6.weight",
                    "classifier.0.bias": "feature_extract.fc6.bias",
                    "classifier.3.weight": "feature_extract.fc7.weight",
                    "classifier.3.bias": "feature_extract.fc7.bias",
                }

                p_dict = torch.load(model_path)
                s_dict = self.state_dict()
                print(s_dict.keys())
                for name in p_dict:
                    if name not in corresp_name:
                        continue
                    if corresp_name[name] in s_dict and s_dict[corresp_name[name]].shape==p_dict[name].shape:
                        print('load weight:',corresp_name[name])
                        s_dict[corresp_name[name]] = p_dict[name]
                self.load_state_dict(s_dict)
            self.layer_shape = {
                2: (128, self.clip_n // 1, 13, 24),
                3: (256, self.clip_n // 2, 6, 12),
                4: (512, self.clip_n // 4, 3, 6),
                5: (512, self.clip_n // 8, 2, 4),
            }


        self.shape_red = nn.AdaptiveMaxPool3d((self.clip_n // 8, 2, 4 if self.backbone == 'c3d' else 3 ))

        if self.use_res:
            self.fc_dim = (self.layer_shape[self.layer_num][0] + self.modal_count * self.transform_dim) * (
                        self.clip_n // 8) * self.layer_shape[5][2] * self.layer_shape[5][3]
        else:
            if self.use_res:
                self.fc_dim = self.layer_shape[self.layer_num][0] * (self.clip_n) * self.layer_shape[5][2] * \
                              self.layer_shape[5][3]
            else:
                self.fc_dim = self.layer_shape[self.layer_num][0] * (self.clip_n // 8) * self.layer_shape[5][2] * self.layer_shape[5][3]

        self.fc7 = nn.Linear(self.fc_dim, 2048)
        self.fc8 = nn.Linear(2048, self.params['class_num'])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        if self.use_res:
            self.adaptpooling = nn.AdaptiveAvgPool3d((self.layer_shape[self.layer_num][1],
                                                      self.layer_shape[self.layer_num][2],
                                                      self.layer_shape[self.layer_num][3]))

    def forward(self, frame, flow, seg, saliency):
        B, C, N, H, W = frame.shape

        merge_features, attention = self.merge_module(frame, flow, seg, saliency)
        print(merge_features.shape)
        if self.use_attention:
            features = attention * merge_features
        else:
            features = merge_features

        features = self.feature_extract(features)
        if self.use_res:
            x2 = self.adaptpooling(merge_features)
            features = torch.concat([features, x2], dim=1)
        features = self.shape_red(features)

        features=features.view(B, -1)
        x = self.relu(self.fc7(features))
        x = self.dropout(x)
        logits = self.fc8(x)
        return logits


