import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")
        
        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        conv_counter = 0
        for out_channels in self.channels:
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **self.conv_params))
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))
            in_channels = out_channels
            conv_counter += 1
            
            if conv_counter == self.pool_every:
                conv_counter = 0
                layers.append(POOLINGS[self.pooling_type](**self.pooling_params))
        
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        result = 0
        t = torch.randn(1, *self.in_size)
        try:
            result = self.feature_extractor(t).flatten().shape[0]
        
        finally:
            torch.set_rng_state(rng_state)
            
        return result

    def _make_mlp(self):
        
        
        dims = self.hidden_dims.copy()
        dims.append(self.out_classes)
        nonlins = [ACTIVATIONS[self.activation_type](**self.activation_params)] * len(self.hidden_dims)
        nonlins.append("none")
        return MLP(self._n_features(), dims, nonlins)

    def forward(self, x: Tensor):

        features = self.feature_extractor(x)
        
        return self.mlp(features.flatten(start_dim=1))


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        layers = []
        shortcut_layers = []
        tmp = in_channels

        for i, out_channels in enumerate(channels):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_sizes[i], padding=kernel_sizes[i] // 2, bias=True))
            if (i < len(channels) - 1) and dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
            if (i < len(channels) - 1) and batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            if i < len(channels) - 1:
                layers.append(ACTIVATIONS[activation_type](**activation_params))

            in_channels = out_channels
            
        if tmp != channels[-1]:
            shortcut_layers.append(nn.Conv2d(tmp, channels[-1], kernel_size=1, bias=False))

        self.main_path = nn.Sequential(*layers)
        self.shortcut_path = nn.Sequential(*shortcut_layers)            

    def forward(self, x: Tensor):
        main = self.main_path(x)
        short = self.shortcut_path(x)
    
        return torch.relu(main + short)


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. NOT the outer projections)
            The length determines the number of convolutions, EXCLUDING the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->5->10.
            The first and last arrows are the 1X1 projection convolutions, 
            and the middle one is the inner convolution (corresponding to the kernel size listed in "inner kernel sizes").
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        inner_channels = [inner_channels[0]] + inner_channels + [in_out_channels]
        inner_kernel_sizes = [1] + inner_kernel_sizes + [1]

        super().__init__(in_out_channels, inner_channels, inner_kernel_sizes, **kwargs)



class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        layers = []
        all_channels = [in_channels, *self.channels]

        for i, out_channels in enumerate(self.channels):
            if i % self.pool_every != 0:
                continue
            
            l, r = i, min(i + self.pool_every, len(self.channels))
            if self.bottleneck and in_channels == self.channels[l:r][-1]:
                layers.append(ResidualBottleneckBlock(
                        in_out_channels = in_channels,
                        inner_channels = self.channels[l:r][1:-1],
                        inner_kernel_sizes = [3]*len(self.channels[l:r][1:-1]),
                        batchnorm=self.batchnorm,
                        dropout=self.dropout,
                        activation_type=self.activation_type,
                        activation_params=self.activation_params))
            else:
                layers.append(ResidualBlock(
                        in_channels=in_channels, 
                        channels=self.channels[l:r], 
                        kernel_sizes=[3]*len(self.channels[l:r]),
                        batchnorm=self.batchnorm,
                        dropout=self.dropout, 
                        activation_type=self.activation_type, 
                        activation_params=self.activation_params))

            in_channels = self.channels[l:r][-1]
            
            if i <= len(self.channels) - self.pool_every:
                layers.append(POOLINGS[self.pooling_type](**self.pooling_params))
    
        return nn.Sequential(*layers)
