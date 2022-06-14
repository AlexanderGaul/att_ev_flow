import torch

from ERAFT.model.extractor import ResidualBlock


class Upsampler(torch.nn.Module) :
    def __init__(self, in_channels, out_channels, norm_fn='batch',
                 num_layers=3) :
        super().__init__()

        self.num_layers = num_layers
        assert self.num_layers == 2 or self.num_layers == 3

        self.conv1 = torch.nn.ConvTranspose2d(in_channels=in_channels,
                                     out_channels=in_channels,
                                     kernel_size=4,
                                     stride=2)

        self.conv2 = torch.nn.ConvTranspose2d(in_channels=in_channels,
                                     out_channels=in_channels // 2 if self.num_layers == 3 else out_channels,
                                     kernel_size=4,
                                     stride=2)
        if  norm_fn == 'batch' :
            self.norm1 = torch.nn.BatchNorm2d(in_channels)
            self.norm2 = torch.nn.BatchNorm2d(in_channels // 2)
        elif norm_fn == 'instance' :
            self.norm1 = torch.nn.InstanceNorm2d(in_channels)
            self.norm2 = torch.nn.InstanceNorm2d(in_channels // 2)
        elif norm_fn == 'none' :
            self.norm1 = torch.nn.Sequential()
            self.norm2 = torch.nn.Sequential()

        self.conv3 = torch.nn.ConvTranspose2d(in_channels=in_channels // 2,
                                     out_channels=out_channels,
                                     kernel_size=4,
                                     stride=2)

    def forward(self, input) :
        x = input
        x = torch.nn.functional.relu(self.norm1(self.conv1(x)[..., 1:-1, 1:-1]))
        x = self.norm2(self.conv2(x)[..., 1:-1, 1:-1])
        if self.num_layers == 3 :
            x = torch.nn.functional.relu(x)
            x = (self.conv3(x)[..., 1:-1, 1:-1])
        return x


class UpsamplerV2(torch.nn.Module) :
    def __init__(self, in_channels, out_channels, norm_fn='none', num_layers=3) :
        super().__init__()
        assert num_layers == 2 or num_layers == 3
        self.num_layers = num_layers

        # TODO: with residual blocks in-between
        self.conv1 = torch.nn.ConvTranspose2d(in_channels=in_channels,
                                              out_channels=in_channels,
                                              kernel_size=4,
                                              stride=2)
        self.res1 = ResidualBlock(in_channels, in_channels, norm_fn=norm_fn)

        self.conv2 = torch.nn.ConvTranspose2d(in_channels=in_channels,
                                              out_channels=in_channels//2,
                                              kernel_size=4,
                                              stride=2)
        self.res2 = ResidualBlock(in_channels//2,
                                  in_channels//2,
                                  norm_fn=norm_fn)

        self.conv3 = torch.nn.ConvTranspose2d(in_channels=in_channels // 2,
                                              out_channels=in_channels // 2,
                                              kernel_size=4,
                                              stride=2)
        self.res3 = ResidualBlock(in_channels // 2, in_channels //2,
                                  norm_fn=norm_fn)

        self.conv_out = torch.nn.Conv2d(in_channels=in_channels // 2,
                                        out_channels=out_channels,
                                        kernel_size=1)


        if  norm_fn == 'batch' :
            self.norm1 = torch.nn.BatchNorm2d(in_channels)
            self.norm2 = torch.nn.BatchNorm2d(in_channels // 2)
            self.norm3 = torch.nn.BatchNorm2d(in_channels // 2)
        elif norm_fn == 'instance' :
            self.norm1 = torch.nn.InstanceNorm2d(in_channels)
            self.norm2 = torch.nn.InstanceNorm2d(in_channels // 2)
            self.norm3 = torch.nn.InstanceNorm2d(in_channels // 2)
        elif norm_fn == 'none' :
            self.norm1 = torch.nn.Sequential()
            self.norm2 = torch.nn.Sequential()
            self.norm3 = torch.nn.Sequential()

    def forward(self, input) :
        x = input
        x = torch.nn.functional.relu(self.norm1(self.conv1(x)[..., 1:-1, 1:-1]))
        x = self.res1(x)
        x = torch.nn.functional.relu(self.norm2(self.conv2(x)[..., 1:-1, 1:-1]))
        x = self.res2(x)
        if self.num_layers == 3:
            x = torch.nn.functional.relu(self.norm3(self.conv3(x)[..., 1:-1, 1:-1]))
            x = self.res3(x)

        return self.conv_out(x)


class BasicEncoder(torch.nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, n_first_channels=1,
                 num_layers=3) :
        assert num_layers == 3 or num_layers == 2
        self.num_layers = num_layers
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = torch.nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = torch.nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = torch.nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = torch.nn.Sequential()

        self.conv1 = torch.nn.Conv2d(n_first_channels, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = torch.nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = torch.nn.Conv2d(128 if self.num_layers == 3 else 96,
                                     output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = torch.nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d, torch.nn.GroupNorm)):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.num_layers == 3 :
            x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x