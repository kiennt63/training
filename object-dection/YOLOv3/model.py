import torch
import torch.nn as nn
from torch.nn.modules import conv

# Tuple: (out_channels, k_size, stride)
# List: ["B", num_repeats] - number of repeats for dense block 
# S: Branch
# U: Upsample

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1), # TODO:
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=(not batch_norm), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.1)
        self.use_batch_norm = batch_norm

    def forward(self, x):
        if self.use_batch_norm:
            return self.lrelu(self.bn(self.conv(x)))

        else: 
            return self.conv(x)

class residual_block(nn.Module):
    def __init__(self, channels, use_residual, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers.append(
                nn.Sequential(
                conv_block(channels, channels // 2, kernel_size=1),
                conv_block(channels // 2, channels, kernel_size=3, padding=1)
                )
            )

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)

        return x

class scale_prediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.prediction = nn.Sequential(
            conv_block(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            conv_block(2 * in_channels, 3 * (num_classes + 5), batch_norm=False, kernel_size=1)
        )

        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.prediction(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
        # N * 3 * (13 * scale) * (13 * scale) * (5 + num_classes)

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, scale_prediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, residual_block):
                if layer.num_repeats == 8:
                    route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()

        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    conv_block(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=(kernel_size // 2)
                    )   
                )

                in_channels = out_channels
            
            elif isinstance(module, list):
                _, num_repeats = module
                layers.append(
                    residual_block(
                        in_channels, 
                        True, 
                        num_repeats
                    )
                )

            elif module == "S":
                layers += [
                    residual_block(in_channels, use_residual=False, num_repeats=1),
                    conv_block(in_channels, in_channels // 2, kernel_size=1),
                    scale_prediction(in_channels // 2, num_classes=self.num_classes)
                ]

                in_channels = in_channels // 2
                
            elif module == "U":
                layers.append(nn.Upsample(scale_factor=2))
                in_channels = in_channels * 3 # concatenate with previous layer with the same spatial dimemsion
        
        return layers


if __name__ == "__main__":

    num_classes = 20
    IMAGE_SIZE = 832
    model = YOLOv3(3, num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))

    out = model(x)

    print(out[2].shape)