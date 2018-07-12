import torch
import torch.nn as nn
import modules 


def f(in_channels, out_channels, hidden_channels):
    return nn.Sequential(
        modules.Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=True),
        modules.Conv2d(hidden_channels, hidden_channels, kernel_size=[1, 1]), nn.ReLU(inplace=True),
        modules.Conv2dZeros(hidden_channels, out_channels))


class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev : (modules.Permute2d.reverse(z), logdet),
        "shuffle": lambda obj, z, logdet, rev : (modules.Permute2d.shuffle(z), logdet),
        "invconv1x1": lambda obj, z, logdet, rev : obj.invconv1x1(z, logdet, rev)
    }
    
    def __init__(self, in_channels, hidden_channels,
                       actnorm_scale=1.0, logscale_factor=3.0,
                       flow_permutation="invconv1x1",
                       LU_decomposed=False,
                       flow_coupling="additive"):
        assert flow_coupling in FlowStep.FlowCoupling,\
            "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)
        assert flow_permutation in FlowStep.FlowPermutation,\
            "float_permutation should be in `{}`".format(FlowStep.FlowPermutation)
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        # 1. actnorm
        self.actnorm = modules.ActNorm2d(in_channels, actnorm_scale, logscale_factor)
        self.f = f(in_channels, hidden_channels, in_channels)
        # 2. permute
        if flow_permutation == "invconv1x1":
            self.invconv1x1 = modules.InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
        # 3. coupling nothing to do now

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        return z, logdet
    
    def reverse_flow(self, input, logdet):
        z, logdet = self.actnorm(input, logdet=logdet, reverse=True)
        return z, logdet
