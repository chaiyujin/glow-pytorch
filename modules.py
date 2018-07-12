import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class _ActNorm(nn.Module):
    """
    Activation Normalization
    
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """
    def __init__(self, num_features, scale=1., logscale_factor=3.):
        super().__init__()
        # register mean and scale
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_features)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(num_features)))
        self.num_features = num_features
        self.logdet_factor = None
        self.scale = scale
        self.logscale_factor = logscale_factor

    def _check_input_dim(self, input):
        return NotImplemented
    
    def initialize_parameters(self, input):
        self._check_input_dim(input)
        assert self.training
        assert input.device == self.bias.device
        with torch.no_grad():
            input = input.transpose(1, -1).contiguous().view(-1, self.num_features)
            mean = torch.mean(input, dim=0, keepdim=False) * -1.0  # reverse
            var  = torch.mean(input ** 2, dim=0, keepdim=False)
            logs = torch.log(self.scale/(torch.sqrt(var)+1e-6)) / self.logscale_factor
            self.bias.data.copy_(mean.data)
            self.logs.data.copy_(logs.data)

    def _center(self, input, reverse=False):
        if not reverse:
            return input + self.bias
        else:
            return input - self.bias

    def _scale(self, input, logdet=None, reverse=False):
        logs = self.logs * self.logscale_factor
        if not reverse:
            input = input * torch.exp(logs)
        else:
            input = input * torch.exp(-logs)

        if logdet is not None:
            dlogdet = torch.sum(logs) * self.logdet_factor
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        self._check_input_dim(input)
        input_size = input.size()
        # set logdet_factor
        self.logdet_factor = np.prod([int(input_size[i]) for i in range(2, len(input_size))])
        
        # transpose feature to last dim
        input = input.transpose(1, -1).contiguous()
        input_size = input.size()  # record current shape
        input = input.view(-1, self.num_features)
        if not reverse:
            # center and scale
            input = self._center(input, reverse)
            # input, logdet = self._scale(input, logdet, reverse)
        else:
            # scale and center
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        # reshape and transpose back
        input = input.view(*input_size).transpose(-1, 1).contiguous()
        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1., logscale_factor=3.):
        super().__init__(num_features, scale, logscale_factor)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`")


class LinearZeros(nn.Linear):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__(in_channels, out_channels)
        self.logscale_factor = logscale_factor
        # set logs parameter
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Conv2d(nn.Conv2d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }
    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels, 
                       kernel_size=[3, 3], stride=[1, 1],
                       padding="same", do_actnorm=True, weight_std=0.05):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)
        self.do_actnorm = do_actnorm
        
    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x = self.actnorm(x)
        return x


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                       kernel_size=[3, 3], stride=[1, 1],
                       padding="same", logscale_factor=3):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Permute2d:
    @staticmethod
    def reverse(input):
        return input[:, ::-1, :, :]

    @staticmethod
    def shuffle(input):
        idx = np.arange(input.size(1))
        np.random.shuffle(idx)
        return input[:, idx, :, :]


class InvertibleConv1x1(nn.Module):
    def __init__(self, in_channels, LU_decomposed=False):
        super().__init__()
        if not LU_decomposed:
            w_shape = [in_channels, in_channels]
            self.w_shape = w_shape
            self.logdet_factor = w_shape[-1] * w_shape[-2]
            # Sample a random orthogonal matrix:
            w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            raise NotImplementedError()

    def forward(self, input, logdet, reverse=False):
        assert logdet is not None
        dlogdet = torch.log(torch.abs(torch.det(self.weight))) * self.logdet_factor
        if not reverse:
            weight = self.weight.view(self.w_shape[0], self.w_shape[1], 1, 1)
            z = F.conv2d(input, weight)
            return z, logdet + dlogdet
        else:
            weight = self.weight.inverse().view(self.w_shape[0], self.w_shape[1], 1, 1)
            z = F.conv2d(input, weight)
            return z, logdet - dlogdet


class Squeeze2d(nn.Module):
    def __init__(self, factor):
        assert factor >= 1 and isinstance(factor, int)
        super().__init__()
        self.factor = factor

    def forward(self, input):
        if self.factor == 1:
            return input
        factor = self.factor
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert H % factor == 0 and W % factor == 0
        x = input.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H // factor, factor, W // factor, factor, C)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, H // factor, W // factor, C * factor * factor).permute(0, 3, 1, 2).contiguous()
        return x


class Unsqueeze2d(nn.Module):
    def __init__(self, factor):
        assert factor >= 1 and isinstance(factor, int)
        super().__init__()
        self.factor = factor

    def forward(self, input):
        if self.factor == 1:
            return input
        factor = self.factor
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert C % (factor ** 2) == 0
        x = input.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H, W, C // (factor ** 2), factor, factor)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, H * factor, W * factor, C // (factor ** 2)).permute(0, 3, 1, 2).contiguous()
        return x


def test_actnorm():
    actnorm = ActNorm2d(16)
    x = torch.Tensor(np.random.rand(2, 16, 4, 4))
    actnorm.initialize_parameters(x)
    y, _ = actnorm(x)
    y = y.permute(0, 2, 3, 1).contiguous().view(-1, 16)
    y = y.mean(dim=0)
    print(y.size())
    print(y)

    conv2d = Conv2dZeros(16, 5)
    y = conv2d(x)
    print(y.size())
    print("conv2d weight", conv2d.weight.size())

    # test squeeze
    print("squeeze")
    squeeze = Squeeze2d(2)
    unsquee = Unsqueeze2d(2)

    print(x.size())
    x_ = squeeze(x)
    print(x_.size())
    x_r = unsquee(x_)
    print(x_r.size())
    print(torch.max(torch.abs(x_r - x)))

    conv1x1 = InvertibleConv1x1(16)
    x_, logdet = conv1x1(x, 0)
    print(x_.size())
    print(logdet.size())


if __name__ == "__main__":
    test_actnorm()
