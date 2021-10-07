import torch
import torch.nn as nn

arch = [
    (7, 64, 2, 3),
    'M',
    (3, 192, 1, 1),
    'M',
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    'M',
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    'M',
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.arch = arch
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.arch)
        self.fcs = self.create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def create_conv_layers(self, arch):
        layers = []
        in_channels = self.in_channels

        for layer in arch:
            if type(layer) == tuple:
                layers += [
                    conv_block(in_channels, layer[1], kernel_size=layer[0], stride=layer[2], padding=layer[3])
                    ]
                in_channels = layer[1]

            elif type(layer) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(layer) == list:
                num_repeat = layer[2]
                for _ in range(num_repeat):
                    for i in range(len(layer) - 1):
                        layers += [
                            conv_block(in_channels, layer[i][1], kernel_size=layer[i][0], stride=layer[i][2], padding=layer[i][3])
                            ]
                        in_channels = layer[i][1]

        return nn.Sequential(*layers)

    def create_fcs(self, grid_size, num_boxes, num_classes):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * grid_size * grid_size, 4096),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, grid_size * grid_size * (num_classes + num_boxes * 5))
        )


if __name__ == '__main__':
    model = Yolov1(grid_size=7, num_boxes=2, num_classes=20)
    x = torch.randn(32, 3, 448, 448)
    print(model(x).shape)