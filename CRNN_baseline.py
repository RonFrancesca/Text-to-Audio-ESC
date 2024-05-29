"""
CRNN baseline taken from
- Castorena, C., Cobos, M., Lopez-Ballester, J., & Ferri, F. J. (2023). A safety-oriented framework for sound event detection in driving scenarios. Applied Acoustics, 215, 109719.
- Ronchini, F., Serizel, R., Turpault, N., & Cornell, S. (2021). The impact of non-target events in synthetic soundscapes for sound event detection. arXiv preprint arXiv:2109.14061.
"""

import torch


class ConvBlockCRNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, do_pool=True):
        super(ConvBlockCRNN, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.GLU = torch.nn.GLU()
        self.pool = torch.nn.AvgPool2d(kernel_size=pool_size)
        self.do_pool = do_pool

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.concatenate(
            (x, x), dim=-1
        )  # N.B. double check, but original baseline uses same tensor as mask in GLU https://github.com/DCASE-REPO/DESED_task/blob/f7dc296e39b09ef26bd05f27b1c720cbc6208bb1/desed_task/nnet/CNN.py#L5
        x = self.GLU(x)
        if self.do_pool:
            x = self.pool(x)
        return x


class CRNNBaseline(torch.nn.Module):

    def __init__(
        self,
        n_mels,
        kernel_size=3,
        filters=[16, 32, 64, 128, 128, 128, 128],
        pool_sizes=[(2, 2), (2, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)],
    ):
        super(CRNNBaseline, self).__init__()

        if n_mels == 64:
            do_pool_last = False
        if n_mels == 128:
            do_pool_last = True

        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_sizes
        # Conv Blocks
        self.convblock1 = ConvBlockCRNN(
            in_channels=1,
            out_channels=filters[0],
            kernel_size=kernel_size,
            pool_size=self.pool_size[0],
        )
        self.convblock2 = ConvBlockCRNN(
            in_channels=filters[0],
            out_channels=filters[1],
            kernel_size=kernel_size,
            pool_size=self.pool_size[1],
        )
        self.convblock3 = ConvBlockCRNN(
            in_channels=filters[1],
            out_channels=filters[2],
            kernel_size=kernel_size,
            pool_size=self.pool_size[2],
        )
        self.convblock4 = ConvBlockCRNN(
            in_channels=filters[2],
            out_channels=filters[3],
            kernel_size=kernel_size,
            pool_size=self.pool_size[3],
        )
        self.convblock5 = ConvBlockCRNN(
            in_channels=filters[3],
            out_channels=filters[4],
            kernel_size=kernel_size,
            pool_size=self.pool_size[4],
        )
        self.convblock6 = ConvBlockCRNN(
            in_channels=filters[4],
            out_channels=filters[5],
            kernel_size=kernel_size,
            pool_size=self.pool_size[5],
        )
        self.convblock7 = ConvBlockCRNN(
            in_channels=filters[5],
            out_channels=filters[6],
            kernel_size=kernel_size,
            pool_size=self.pool_size[6],
            do_pool=do_pool_last,
        )

        # Bidirectional GRU
        self.GRU = torch.nn.GRU(
            input_size=128, hidden_size=128, num_layers=2, bidirectional=True
        )

        # Now let's do the classification head
        self.flatten = torch.nn.Flatten()
        self.dense = torch.nn.Linear(in_features=2816, out_features=10)

    def forward(self, x):

        # INPUT SHOULD HAVE SIZE: BATCH X CHANNEL X TIME X FREQ (so we permute input tensor)
        x_input = torch.permute(
            x, (0, 1, 3, 2)
        )  # Luca: N.B. salamon applies 4X2 pooling over TXF axis, but specs are returned in FXT form, so the reshaping

        x = self.convblock1(x_input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)

        # Output at this point must be BATCH X TIME X FREQ X CHANNEL
        x = torch.permute(x, dims=(0, 2, 1, 3)).squeeze(-1)

        # Let's apply bidirectional GRU
        x, _ = self.GRU(x)

        # Here is different from the paper N.B. the original baseline gives classification
        # Output per time instant, here we want to give a single classification output for the
        # whole audio track
        x = self.flatten(x)
        logits = self.dense(x)

        return logits
