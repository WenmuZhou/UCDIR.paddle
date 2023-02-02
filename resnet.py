import paddle

from typing import Type, Any, Callable, Union, List, Optional
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2',
    'wide_resnet101_2']

__dict__ = {x:x for x in __all__}

model_urls = {'resnet18':
    'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34':
    'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152':
    'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d':
    'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d':
    'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2':
    'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2':
    'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'}


def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1,
    dilation: int=1) ->paddle.nn.Conv2D:
    """3x3 convolution with padding"""
    return paddle.nn.Conv2D(in_channels=in_planes, out_channels=out_planes,
        kernel_size=3, stride=stride, padding=dilation, groups=groups,
        dilation=dilation, bias_attr=False)


def conv1x1(in_planes: int, out_planes: int, stride: int=1) ->paddle.nn.Conv2D:
    """1x1 convolution"""
    return paddle.nn.Conv2D(in_channels=in_planes, out_channels=out_planes,
        kernel_size=1, stride=stride, bias_attr=False)


class BasicBlock(paddle.nn.Layer):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int=1,
        downsample: Optional[paddle.nn.Layer]=None, groups: int=1,
        base_width: int=64, dilation: int=1, norm_layer: Optional[Callable[
        ..., paddle.nn.Layer]]=None) ->None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = paddle.nn.BatchNorm2D
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = paddle.nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        """Tensor Attribute: torch.Tensor.stride, not convert, please check whether it is torch.Tensor.* and convert manually"""
        self.stride = stride

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(paddle.nn.Layer):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int=1,
        downsample: Optional[paddle.nn.Layer]=None, groups: int=1,
        base_width: int=64, dilation: int=1, norm_layer: Optional[Callable[
        ..., paddle.nn.Layer]]=None) ->None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = paddle.nn.BatchNorm2D
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = paddle.nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(paddle.nn.Layer):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers:
        List[int], num_classes: int=1000, zero_init_residual: bool=False,
        groups: int=1, width_per_group: int=64,
        replace_stride_with_dilation: Optional[List[bool]]=None, norm_layer:
        Optional[Callable[..., paddle.nn.Layer]]=None) ->None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = paddle.nn.BatchNorm2D
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element tuple, got {}'
                .format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=self.
            inplanes, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = paddle.nn.ReLU()
        self.maxpool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2])
        self.avgpool = paddle.nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc = paddle.nn.Linear(in_features=512 * block.expansion,
            out_features=num_classes)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]],
        planes: int, blocks: int, stride: int=1, dilate: bool=False
        ) ->paddle.nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = paddle.nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self
            .groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer))
        return paddle.nn.Sequential(*layers)

    def _forward_impl(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = paddle.flatten(x=x, start_axis=1)
        x = self.fc(x)
        return x

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        return self._forward_impl(x)


def load_torch_params(paddle_model, torch_patams):
    paddle_params = paddle_model.state_dict()

    fc_names = ['classifier']
    for key,torch_value in torch_patams.items():
        if 'num_batches_tracked' in key:
            continue
        key = key.replace("running_var", "_variance").replace("running_mean", "_mean").replace("module.", "")
        torch_value = torch_value.detach().cpu().numpy()
        if key in paddle_params:
            flag = [i in key for i in fc_names]
            if any(flag) and "weight" in key:  # ignore bias
                new_shape = [1, 0] + list(range(2, torch_value.ndim))
                print(f"name: {key}, ori shape: {torch_value.shape}, new shape: {torch_value.transpose(new_shape).shape}")
                torch_value = torch_value.transpose(new_shape)
            paddle_params[key] = torch_value
        else:
            print(f'{key} not in paddle')
    paddle_model.set_state_dict(paddle_params)

def load_models(model, model_name, progress):
    from torch.hub import load_state_dict_from_url
    torch_patams = load_state_dict_from_url(model_urls[model_name], progress=progress)
    load_torch_params(model, torch_patams)


def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers:
    List[int], pretrained: bool, progress: bool, **kwargs: Any) ->ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        load_models(model, arch, progress=progress)
    return model


def resnet18(pretrained: bool=False, progress: bool=True, **kwargs: Any
    ) ->ResNet:
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained,
        progress, **kwargs)


def resnet34(pretrained: bool=False, progress: bool=True, **kwargs: Any
    ) ->ResNet:
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained,
        progress, **kwargs)


def resnet50(pretrained: bool=False, progress: bool=True, **kwargs: Any
    ) ->ResNet:
    """ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained,
        progress, **kwargs)


def resnet101(pretrained: bool=False, progress: bool=True, **kwargs: Any
    ) ->ResNet:
    """ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained,
        progress, **kwargs)


def resnet152(pretrained: bool=False, progress: bool=True, **kwargs: Any
    ) ->ResNet:
    """ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained,
        progress, **kwargs)


def resnext50_32x4d(pretrained: bool=False, progress: bool=True, **kwargs: Any
    ) ->ResNet:
    """ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained,
        progress, **kwargs)


def resnext101_32x8d(pretrained: bool=False, progress: bool=True, **kwargs: Any
    ) ->ResNet:
    """ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
        pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool=False, progress: bool=True, **kwargs: Any
    ) ->ResNet:
    """Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained,
        progress, **kwargs)


def wide_resnet101_2(pretrained: bool=False, progress: bool=True, **kwargs: Any
    ) ->ResNet:
    """Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
        pretrained, progress, **kwargs)
