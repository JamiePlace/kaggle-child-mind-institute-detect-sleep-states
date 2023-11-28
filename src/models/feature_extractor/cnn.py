from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


# ref: https://github.com/analokmaus/kaggle-g2net-public/tree/main/models1d_pytorch
class CNNSpectrogram(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int | tuple = 128,
        kernel_sizes: tuple = (32, 16, 4, 2),
        stride: int = 4,
        sigmoid: bool = False,
        output_size: Optional[int] = None,
        conv: Callable = nn.Conv1d,
        reinit: bool = True,
    ):
        super().__init__()
        self.out_chans = len(kernel_sizes)
        self.out_size = output_size
        self.sigmoid = sigmoid
        if isinstance(base_filters, int):
            base_filters = tuple([base_filters])
        self.height = base_filters[-1]
        self.spec_conv = nn.ModuleList()
        for i in range(self.out_chans):
            tmp_block = [
                conv(
                    in_channels,
                    base_filters[0],
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    padding=(kernel_sizes[i] - 1) // 2,
                )
            ]
            if len(base_filters) > 1:
                for j in range(len(base_filters) - 1):
                    tmp_block = tmp_block + [
                        nn.BatchNorm1d(base_filters[j]),
                        nn.ReLU(inplace=True),
                        conv(
                            base_filters[j],
                            base_filters[j + 1],
                            kernel_size=kernel_sizes[i],
                            stride=stride,
                            padding=(kernel_sizes[i] - 1) // 2,
                        ),
                    ]
                self.spec_conv.append(nn.Sequential(*tmp_block))
            else:
                self.spec_conv.append(tmp_block[0])

        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (_type_): (batch_size, in_channels, time_steps)

        Returns:
            _type_: (batch_size, out_chans, height, time_steps)
        """
        # x: (batch_size, in_channels, time_steps)
        out: list[torch.Tensor] = []
        for i in range(self.out_chans):
            out.append(self.spec_conv[i](x))
        img = torch.stack(
            out, dim=1
        )  # (batch_size, out_chans, height, time_steps)
        if self.out_size is not None:
            img = self.pool(img)  # (batch_size, out_chans, height, out_size)
        if self.sigmoid:
            img = img.sigmoid()
        return img


class CNNextractor(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_right = nn.ModuleList()
        dropout = 0.5
        dilation_left = 1
        dilation_right = 4
        self.conv_left = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=base_filters,
                kernel_size=11,
                stride=1,
                dilation=1,
                padding="same",
            ),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout),
            nn.Conv1d(
                in_channels=base_filters,
                out_channels=base_filters,
                kernel_size=11,
                stride=1,
                dilation=dilation_left,
                padding="same",
            ),
            nn.Conv1d(
                in_channels=base_filters,
                out_channels=base_filters,
                kernel_size=11,
                stride=1,
                dilation=dilation_left,
                padding="same",
            ),
            nn.Conv1d(
                in_channels=base_filters,
                out_channels=base_filters,
                kernel_size=11,
                stride=1,
                dilation=dilation_left,
                padding="same",
            ),
        )
        self.conv_right = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=base_filters,
                kernel_size=11,
                stride=1,
                dilation=dilation_right,
                padding="same",
            ),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.5),
            nn.Conv1d(
                in_channels=base_filters,
                out_channels=base_filters,
                kernel_size=11,
                stride=1,
                dilation=dilation_right,
                padding="same",
            ),
            nn.Conv1d(
                in_channels=base_filters,
                out_channels=base_filters,
                kernel_size=11,
                stride=1,
                dilation=dilation_right,
                padding="same",
            ),
            nn.Conv1d(
                in_channels=base_filters,
                out_channels=base_filters,
                kernel_size=11,
                stride=1,
                dilation=dilation_right,
                padding="same",
            ),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x torch.Tensor: (batch_size, window_size, n_features)

        Returns:
            Tuple[Tensor, Tensor]: (some number), (batch_size, base_filters, some number)
        """
        x = x.view((x.shape[0], self.in_channels, -1))
        out_left: torch.Tensor = self.conv_left(x)
        out_right: torch.Tensor = self.conv_right(x)
        return torch.cat([out_left, out_right], dim=2).view(
            (x.shape[0], -1)
        ), torch.cat([out_left, out_right], dim=2)


class CNNrefinor(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        base_filters: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        dropout = 0.5
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=base_filters,
                kernel_size=11,
                stride=1,
                dilation=1,
                padding="same",
            ),
            nn.Upsample(scale_factor=2, mode="linear"),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=base_filters,
                kernel_size=11,
                stride=1,
                dilation=1,
                padding="same",
            ),
            nn.Dropout(p=dropout),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x torch.Tensor: (batch_size, window_size, n_features)

        Returns:
            Tuple[Tensor, Tensor]: (some number), (batch_size, base_filters, some number)
        """
        return self.conv(x)
