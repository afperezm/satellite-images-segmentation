from typing import Any

import torch.nn as nn
import torchvision.models as models


def resnet18(num_classes, num_channels):
    """ ResNet-18 model with variable number of input channels and number of classes. """

    # https://discuss.pytorch.org/t/resnet18-output-shape-depending-on-number-of-in-channels/100733

    model = models.resnet18()

    num_out_channels = model.conv1.out_channels
    kernel_size = model.conv1.kernel_size
    stride = model.conv1.stride
    padding = model.conv1.padding

    model.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=False)

    num_in_features = model.fc.in_features

    model.fc = nn.Linear(in_features=num_in_features, out_features=num_classes)

    return model


def resnet50(num_classes, num_channels):
    """ ResNet-18 model with variable number of input channels and number of classes. """

    # https://discuss.pytorch.org/t/resnet18-output-shape-depending-on-number-of-in-channels/100733

    model = models.resnet50()

    num_out_channels = model.conv1.out_channels
    kernel_size = model.conv1.kernel_size
    stride = model.conv1.stride
    padding = model.conv1.padding

    model.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=False)

    num_in_features = model.fc.in_features

    model.fc = nn.Linear(in_features=num_in_features, out_features=num_classes)

    return model
