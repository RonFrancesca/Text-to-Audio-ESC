import torch


class CNNNetwork(torch.nn.Module):

    def __init__(
        self,
        n_mels,
        net_data,
        pool_size=(4, 2),
        pool_stride=(4, 2),
    ):
        super().__init__()

        self.droupout_rate = net_data.dropout_rate
        self.kernel_size = net_data.kernel_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=24, kernel_size=self.kernel_size, padding="same"
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=24,
            out_channels=48,
            kernel_size=self.kernel_size,
            padding="same",
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=48,
            out_channels=48,
            kernel_size=self.kernel_size,
            padding="same",
        )

        self.activation1 = torch.nn.ReLU()
        self.activation2 = torch.nn.ReLU()
        self.activation3 = torch.nn.ReLU()
        self.activation4 = torch.nn.ReLU()

        self.pool1 = torch.nn.MaxPool2d(
            kernel_size=self.pool_size, stride=self.pool_stride
        )
        self.pool2 = torch.nn.MaxPool2d(
            kernel_size=self.pool_size, stride=self.pool_stride
        )

        self.flatten = torch.nn.Flatten()

        # to be removed afterwards
        if n_mels == 128:
            in_features_layer_1 = 3072  # 1680
        elif n_mels == 64:
            in_features_layer_1 = 1536

        self.fc1 = torch.nn.Linear(in_features=in_features_layer_1, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=10)

        self.dr1 = torch.nn.Dropout(p=self.droupout_rate)
        self.dr2 = torch.nn.Dropout(p=self.droupout_rate)

    def forward(self, x):

        x = torch.permute(
            x, (0, 1, 3, 2)
        )  # Luca: N.B. salamon applies 4X2 pooling over TXF axis, but specs are returned in FXT form, so the reshaping

        #
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.activation1(x)

        # cnn layer-2
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.activation2(x)

        # cnn layer-3
        x = self.conv3(x)
        x = self.activation3(x)

        x = self.flatten(x)

        # dense layer-1
        x = self.dr1(x)
        x = self.fc1(x)
        x = self.activation4(x)

        # dense output layer
        x = self.dr2(x)  # N.B. salamon applies dropout before last layer
        logits = self.fc2(x)

        return logits
