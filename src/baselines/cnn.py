import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable


class UNet(nn.Module):
    def __init__(
        self,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        n_channels_max: int = 1024,
        n_channels_first_step: int = 64,
        batch_norm: bool = True,
        activation: Callable = nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.encoder = Encoder(
            n_channels_in, n_channels_max, n_channels_first_step, batch_norm, activation
        )
        self.decoder = Decoder(
            n_channels_max, n_channels_out, n_channels_first_step, batch_norm, activation
        )

    def forward(self, x):
        if x.ndim != 4:
            x = x.unsqueeze(1)
        out, encoder_layers_out = self.encoder(x)
        out = self.decoder(out, encoder_layers_out)
        return out[:, 0, ...]

    def reset_parameters(self, **kwargs):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()


class ConvBlock(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, batch_norm=True, activation=nn.ReLU):
        super().__init__()

        if batch_norm:
            self.conv_block = nn.Sequential(
                nn.Conv2d(n_channels_in, n_channels_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_channels_out),
                activation(),
                nn.Conv2d(n_channels_out, n_channels_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_channels_out),
                activation(),
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(n_channels_in, n_channels_out, kernel_size=3, padding=1),
                activation(),
                nn.Conv2d(n_channels_out, n_channels_out, kernel_size=3, padding=1),
                activation(),
            )

    def forward(self, x):
        return self.conv_block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        n_channels_in,
        n_channels_out,
        n_channels_first_step,
        batch_norm=True,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)

        layers = []
        c_in = n_channels_in
        c_out = n_channels_first_step
        while c_out <= n_channels_out:
            layers.append(ConvBlock(c_in, c_out, batch_norm, activation))
            c_in = c_out
            c_out *= 2
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        encoder_list = []
        out = x
        for l, conv in enumerate(self.layers):
            out = conv(out)
            encoder_list.append(out)
            if l < len(self.layers) - 1:
                out = self.max_pool(out)
        return out, encoder_list[:-1]

    def reset_parameters(self, **kwargs):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class Decoder(nn.Module):
    def __init__(
        self,
        n_channels_in,
        n_channels_out,
        n_channels_first_step,
        batch_norm=True,
        activation=nn.ReLU,
    ):
        super().__init__()

        conv_transpose = []
        conv_blocks = []

        c_in = n_channels_in
        c_out = n_channels_in // 2
        while c_out >= n_channels_first_step:
            conv_transpose.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2))
            conv_blocks.append(ConvBlock(c_in, c_out, batch_norm, activation))
            c_in = c_out
            c_out = c_out // 2

        self.conv_transpose = nn.ModuleList(conv_transpose)
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.conv_out = nn.Conv2d(n_channels_first_step, n_channels_out, kernel_size=1)

    def forward(self, x, encoder_values):
        out = x
        for conv_trans, conv, enc in zip(
            self.conv_transpose, self.conv_blocks, encoder_values[::-1]
        ):
            out = conv_trans(out)

            diffY = enc.size()[2] - out.size()[2]
            diffX = enc.size()[3] - out.size()[3]

            out = F.pad(out, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            out = torch.cat([out, enc], dim=1)
            out = conv(out)

        out = self.conv_out(out)
        return out

    def reset_parameters(self, **kwargs):
        for m in [self.conv_transpose, self.conv_blocks]:
            for layer in m:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        self.conv_out.reset_parameters()
