'''
create by ykk648: 19-12-25
https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe
http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf
'''

import torch
import torch.nn as nn

from . import slim
from .slim import g_name

from .common_utility import get_shuffle_ave_pooling_size
from .common_utility import l2_norm as l2_norm


class BasicBlock(nn.Module):

    def __init__(self, name, in_channels, out_channels, stride, dilation):
        super(BasicBlock, self).__init__()
        self.g_name = name
        self.in_channels = in_channels
        self.stride = stride
        channels = out_channels // 2
        if stride == 1:
            assert in_channels == out_channels
            self.conv = nn.Sequential(
                slim.conv_bn_prelu(name + '/conv1', channels, channels, 1),
                slim.conv_bn(name + '/conv2',
                             channels, channels, 3, stride=stride,
                             dilation=dilation, padding=dilation, groups=channels),
                slim.conv_bn_prelu(name + '/conv3', channels, channels, 1),
            )
        else:
            self.conv = nn.Sequential(
                slim.conv_bn_prelu(name + '/conv1', in_channels, channels, 1),
                slim.conv_bn(name + '/conv2',
                             channels, channels, 3, stride=stride,
                             dilation=dilation, padding=dilation, groups=channels),
                slim.conv_bn_prelu(name + '/conv3', channels, channels, 1),
            )
            self.conv0 = nn.Sequential(
                slim.conv_bn(name + '/conv4',
                             in_channels, in_channels, 3, stride=stride,
                             dilation=dilation, padding=dilation, groups=in_channels),
                slim.conv_bn_prelu(name + '/conv5', in_channels, channels, 1),
            )
        self.shuffle = slim.channel_shuffle(name + '/shuffle', 2)

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.data.shape[1] // 2), :, :]
            x2 = x[:, (x.data.shape[1] // 2):, :, :]
            x = torch.cat((x1, self.conv(x2)), 1)
        else:
            x = torch.cat((self.conv0(x), self.conv(x)), 1)
        return self.shuffle(x)


class Network(nn.Module):

    def __init__(self, num_classes, width_multiplier, input_size):
        super(Network, self).__init__()

        ave_pool_size = get_shuffle_ave_pooling_size(input_size[0], input_size[1])
        width_config = {
            0.25: (24, 48, 96, 512),
            0.33: (32, 64, 128, 512),
            0.5: (48, 96, 192, 1024),
            1.0: (116, 232, 464, 1024),
            1.5: (176, 352, 704, 1024),
            2.0: (244, 488, 976, 2048),
        }
        width_config = width_config[width_multiplier]
        self.num_classes = num_classes
        in_channels = 24

        # outputs, stride, dilation, blocks, type
        self.network_config = [
            g_name('data/bn', nn.BatchNorm2d(3)),
            slim.conv_bn_prelu('stage1/conv', 3, in_channels, 3, 2, 1),

            (width_config[0], 2, 1, 4, 'b'),  # out_channels, stride, dilation, num_blocks, stage_type
            (width_config[1], 2, 1, 8, 'b'),  # x16
            (width_config[2], 2, 1, 4, 'b'),  # x32
            slim.conv_bn_prelu('conv5', width_config[2], width_config[3], 1),
            slim.conv_dw('conv_dw', width_config[3], 1024, (int(ave_pool_size[0]), int(ave_pool_size[1])),
                         groups=width_config[3]),

            # slim.flatten("flatten1", 1),
            # g_name('fc', nn.Linear(1024, self.num_classes, bias=False)),
            # slim.flatten("flatten2", 1),

            g_name('fc', nn.Conv2d(1024, self.num_classes, 1)),
            slim.flatten("fc/flatten", 1),
            g_name('fc/bn', nn.BatchNorm1d(self.num_classes)),
        ]
        self.network = []
        for i, config in enumerate(self.network_config):
            if isinstance(config, nn.Module):
                # if not use_pooling and isinstance(config, nn.MaxPool2d):
                #     continue
                self.network.append(config)
                continue
            out_channels, stride, dilation, num_blocks, stage_type = config
            stage_prefix = 'stage_{}'.format(i - 1)
            blocks = [BasicBlock(stage_prefix + '_1', in_channels,
                                 out_channels, stride, dilation)]
            for i in range(1, num_blocks):
                blocks.append(BasicBlock(stage_prefix + '_{}'.format(i + 1),
                                         out_channels, out_channels, 1, dilation))
            self.network += [nn.Sequential(*blocks)]

            in_channels = out_channels

        self.network = nn.Sequential(*self.network)

        self._initialize_weights()

        # for name, m in self.named_modules():
        #     if any(map(lambda x: isinstance(m, x), [nn.Linear, nn.Conv1d, nn.Conv2d])):
        #         # nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def trainable_parameters(self):
        parameters = [
            {'params': self.cls_head_list.parameters(), 'lr_mult': 1.0},
            {'params': self.loc_head_list.parameters(), 'lr_mult': 1.0},
            # {'params': self.network.parameters(), 'lr_mult': 0.1},
        ]
        for i in range(len(self.network)):
            lr_mult = 0.1 if i in (0, 1, 2, 3, 4, 5) else 1
            parameters.append(
                {'params': self.network[i].parameters(), 'lr_mult': lr_mult}
            )
        return parameters

    def forward(self, x):
        x = self.network(x)
        return l2_norm(x)

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'stage1/conv' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    input = torch.randn(1, 3, 112, 112)
    model = Network(num_classes=512, width_multiplier=1.5, input_size=(112, 112))
    print(model, model.network[8])
    model.eval()
    out = model(input)
    print(out.shape)
