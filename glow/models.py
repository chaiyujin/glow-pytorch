import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from . import thops
from . import modules
from . import utils


def f(in_channels, out_channels, hidden_channels):
    return nn.Sequential(
        modules.Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False),
        modules.Conv2d(hidden_channels, hidden_channels, kernel_size=[1, 1]), nn.ReLU(inplace=False),
        modules.Conv2dZeros(hidden_channels, out_channels))


class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev)
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0,
                 flow_permutation="invconv",
                 flow_coupling="additive",
                 LU_decomposed=False):
        # check configures
        assert flow_coupling in FlowStep.FlowCoupling,\
            "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)
        assert flow_permutation in FlowStep.FlowPermutation,\
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        # 1. actnorm
        self.actnorm = modules.ActNorm2d(in_channels, actnorm_scale)
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = modules.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)
        elif flow_permutation == "shuffle":
            self.shuffle = modules.Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = modules.Permute2d(in_channels, shuffle=False)
        # 3. coupling
        if flow_coupling == "additive":
            self.f = f(in_channels // 2, in_channels // 2, hidden_channels)
        elif flow_coupling == "affine":
            self.f = f(in_channels // 2, in_channels, hidden_channels)

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)
        # 3. coupling
        z1, z2 = thops.split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.f(z1)
        elif self.flow_coupling == "affine":
            h = self.f(z1)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = thops.cat_feature(z1, z2)
        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # 1.coupling
        z1, z2 = thops.split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1)
        elif self.flow_coupling == "affine":
            h = self.f(z1)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = thops.cat_feature(z1, z2)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)
        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, logdet


class FlowNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L,
                 actnorm_scale=1.0,
                 flow_permutation="invconv",
                 flow_coupling="additive",
                 LU_decomposed=False):
        """
                             K                                      K
        --> [Squeeze] -> [FlowStep] -> [Split] -> [Squeeze] -> [FlowStep]
               ^                           v
               |          (L - 1)          |
               + --------------------------+
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        self.L = L
        H, W, C = image_shape
        assert C == 1 or C == 3, ("image_shape should be HWC, like (64, 64, 3)"
                                  "C == 1 or C == 3")
        for i in range(L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(modules.SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])
            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(in_channels=C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation=flow_permutation,
                             flow_coupling=flow_coupling,
                             LU_decomposed=LU_decomposed))
                self.output_shapes.append(
                    [-1, C, H, W])
            # 3. Split2d
            if i < L - 1:
                self.layers.append(modules.Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, input, logdet=0., reverse=False, eps_std=None):
        if not reverse:
            return self.encode(input, logdet)
        else:
            return self.decode(input, eps_std)

    def encode(self, z, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, eps_std=None):
        for layer in reversed(self.layers):
            if isinstance(layer, modules.Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, eps_std=eps_std)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z


class Glow(nn.Module):
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()

    def __init__(self, hparams):
        super().__init__()
        self.flow = FlowNet(image_shape=hparams.Glow.image_shape,
                            hidden_channels=hparams.Glow.hidden_channels,
                            K=hparams.Glow.K,
                            L=hparams.Glow.L,
                            actnorm_scale=hparams.Glow.actnorm_scale,
                            flow_permutation=hparams.Glow.flow_permutation,
                            flow_coupling=hparams.Glow.flow_coupling,
                            LU_decomposed=hparams.Glow.LU_decomposed)
        self.hparams = hparams
        self.y_classes = hparams.Glow.y_classes
        # for prior
        if hparams.Glow.learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top = modules.Conv2dZeros(C * 2, C * 2)
        if hparams.Glow.y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = modules.LinearZeros(
                hparams.Glow.y_classes, 2 * C)
            self.project_class = modules.LinearZeros(
                C, hparams.Glow.y_classes)
        # register prior hidden
        num_device = len(utils.get_proper_device(hparams.Device.glow, False))
        assert hparams.Train.batch_size % num_device == 0
        self.register_parameter(
            "prior_h",
            nn.Parameter(torch.zeros([hparams.Train.batch_size // num_device,
                                      self.flow.output_shapes[-1][1] * 2,
                                      self.flow.output_shapes[-1][2],
                                      self.flow.output_shapes[-1][3]])))

    def prior(self, y_onehot=None):
        B, C = self.prior_h.size(0), self.prior_h.size(1)
        h = self.prior_h.detach().clone()
        assert torch.sum(h) == 0.0
        if self.hparams.Glow.learn_top:
            h = self.learn_top(h)
        if self.hparams.Glow.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot).view(B, C, 1, 1)
            h += yp
        return thops.split_feature(h, "split")

    def forward(self, x=None, y_onehot=None, z=None,
                eps_std=None, reverse=False):
        if not reverse:
            return self.normal_flow(x, y_onehot)
        else:
            return self.reverse_flow(z, y_onehot, eps_std)

    def normal_flow(self, x, y_onehot):
        pixels = thops.pixels(x)
        z = x + torch.normal(mean=torch.zeros_like(x),
                             std=torch.ones_like(x) * (1. / 256.))
        logdet = torch.zeros_like(x[:, 0, 0, 0])
        logdet += float(-np.log(256.) * pixels)
        # encode
        z, objective = self.flow(z, logdet=logdet, reverse=False)
        # prior
        mean, logs = self.prior(y_onehot)
        objective += modules.GaussianDiag.logp(mean, logs, z)

        if self.hparams.Glow.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # return
        nll = (-objective) / float(np.log(2.) * pixels)
        return z, nll, y_logits

    def reverse_flow(self, z, y_onehot, eps_std):
        with torch.no_grad():
            mean, logs = self.prior(y_onehot)
            if z is None:
                z = modules.GaussianDiag.sample(mean, logs, eps_std)
            x = self.flow(z, eps_std=eps_std, reverse=True)
        return x

    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if (m.__class__.__name__.find("ActNorm") >= 0):
                m.inited = inited

    def generate_z(self, img):
        self.eval()
        B = self.hparams.Train.batch_size
        x = img.unsqueeze(0).repeat(B, 1, 1, 1).cuda()
        z,_, _ = self(x)
        self.train()
        return z[0].detach().cpu().numpy()

    def generate_attr_deltaz(self, dataset):
        assert "y_onehot" in dataset[0]
        self.eval()
        with torch.no_grad():
            B = self.hparams.Train.batch_size
            N = len(dataset)
            attrs_pos_z = [[0, 0] for _ in range(self.y_classes)]
            attrs_neg_z = [[0, 0] for _ in range(self.y_classes)]
            for i in tqdm(range(0, N, B)):
                j = min([i + B, N])
                # generate z for data from i to j
                xs = [dataset[k]["x"] for k in range(i, j)]
                while len(xs) < B:
                    xs.append(dataset[0]["x"])
                xs = torch.stack(xs).cuda()
                zs, _, _ = self(xs)
                for k in range(i, j):
                    z = zs[k - i].detach().cpu().numpy()
                    # append to different attrs
                    y = dataset[k]["y_onehot"]
                    for ai in range(self.y_classes):
                        if y[ai] > 0:
                            attrs_pos_z[ai][0] += z
                            attrs_pos_z[ai][1] += 1
                        else:
                            attrs_neg_z[ai][0] += z
                            attrs_neg_z[ai][1] += 1
                # break
            deltaz = []
            for ai in range(self.y_classes):
                if attrs_pos_z[ai][1] == 0:
                    attrs_pos_z[ai][1] = 1
                if attrs_neg_z[ai][1] == 0:
                    attrs_neg_z[ai][1] = 1
                z_pos = attrs_pos_z[ai][0] / float(attrs_pos_z[ai][1])
                z_neg = attrs_neg_z[ai][0] / float(attrs_neg_z[ai][1])
                deltaz.append(z_pos - z_neg)
        self.train()
        return deltaz

    @staticmethod
    def loss_generative(nll):
        # Generative loss
        return torch.mean(nll)

    @staticmethod
    def loss_multi_classes(y_logits, y_onehot):
        if y_logits is None:
            return 0
        else:
            return Glow.BCE(y_logits, y_onehot.float())

    @staticmethod
    def loss_class(y_logits, y):
        if y_logits is None:
            return 0
        else:
            return Glow.CE(y_logits, y.long())
