import torch


class ResidualBlock3D(torch.nn.Module) :
    def __init__(self, in_planes, planes, norm_fn='group', stride=1) :
        super().__init__()
        self.conv1 = torch.nn.Conv3d(in_planes, planes,
                                     kernel_size=(3, 3, 3),
                                     padding=1,
                                     stride=(1, stride, stride))
        self.conv2 = torch.nn.Conv3d(planes, planes,
                                     kernel_size=(3, 3, 3),
                                     padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        if norm_fn == 'batch' :
            self.norm1 = torch.nn.BatchNorm3d(planes)
            self.norm2 = torch.nn.BatchNorm3d(planes)
            if not stride == 1:
                self.norm3 = torch.nn.BatchNorm3d(planes)
        elif norm_fn == 'instance' :
            self.norm1 = torch.nn.InstanceNorm3d(planes)
            self.norm2 = torch.nn.InstanceNorm3d(planes)
            if not stride == 1 :
                self.norm3 = torch.nn.InstanceNorm3d(planes)
        elif norm_fn == 'none' :
            self.norm1 = torch.nn.Sequential()
            self.norm2 = torch.nn.Sequential()
            if not stride == 1 :
                self.norm3 = torch.nn.Sequential()

        if stride == 1 :
            self.downsample = None
        else :
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv3d(in_planes, planes,
                                kernel_size=1,
                                stride=(1, stride, stride)))

    def forward(self, x) :
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None :
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder3D(torch.nn.Module) :
    def __init__(self, output_dim, norm_fn='batch', n_first_channels=1, num_layers=3) :
        super().__init__()
        self.norm_fn = norm_fn
        assert num_layers == 3 or num_layers == 2
        self.num_layers = num_layers

        if self.norm_fn == 'batch':
            self.norm1 = torch.nn.BatchNorm3d(64)
        elif self.norm_fn == 'instance' :
            self.norm1 = torch.nn.InstanceNorm3d(64)
        elif self.norm_fn == 'none' :
            self.norm1 = torch.nn.Sequential()

        self.conv1 = torch.nn.Conv3d(n_first_channels, 64,
                                     kernel_size=(3, 7, 7),
                                     stride=(1, 2, 2),
                                     padding=(1, 3, 3))
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output_convolution
        self.conv2 = torch.nn.Conv3d(128 if self.num_layers == 3 else 96,
                                     output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm3d, torch.nn.InstanceNorm3d, torch.nn.GroupNorm)):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)


    def _make_layer(self, dim, stride=1) :
        layer1 = ResidualBlock3D(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock3D(dim, dim, self.norm_fn, stride=1)
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
        """
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        """
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x