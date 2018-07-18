"""
Test the modules
"""
import cv2
import torch
import tensorflow as tf
import numpy as np
from glow import thops
from glow import modules
from glow import models


def is_equal(a, b, eps=1e-5):
    if a.shape != b.shape:
        return False
    max_delta = np.max(np.abs(a - b))
    return max_delta < eps


def test_multidim_sum():
    x = np.random.rand(2, 3, 4, 4)
    th_x = torch.Tensor(x)
    tf_x = tf.constant(x)
    test_axis_list = [[1], [1, 2], [0, 2, 3], [0, 1, 2, 3]]
    with tf.Session():
        print("[Test] multidim sum, compared with tensorflow")
        for axis in test_axis_list:
            for keep in [False, True]:
                # tf
                tf_y = tf.reduce_sum(tf_x, axis=axis, keepdims=keep)
                tf_y = tf_y.eval()
                # th
                th_y = thops.sum(th_x, dim=axis, keepdim=keep).numpy()
                if is_equal(th_y, tf_y):
                    print("  Pass: dim={}, keepdim={}", axis, keep)
                else:
                    raise ValueError("sum with dim={} error".format(axis))


def test_multidim_mean():
    x = np.random.rand(2, 3, 4, 4)
    th_x = torch.Tensor(x)
    tf_x = tf.constant(x)
    test_axis_list = [[1], [1, 2], [0, 2, 3], [0, 1, 2, 3]]
    with tf.Session():
        print("[Test] multidim mean, compared with tensorflow")
        for axis in test_axis_list:
            for keep in [False, True]:
                # tf
                tf_y = tf.reduce_mean(tf_x, axis=axis, keepdims=keep)
                tf_y = tf_y.eval()
                # th
                th_y = thops.mean(th_x, dim=axis, keepdim=keep).numpy()
                if is_equal(th_y, tf_y):
                    print("  Pass: dim={}, keepdim={}", axis, keep)
                else:
                    raise ValueError("mean with dim={} error".format(axis))


def test_actnorm():
    print("[Test]: actnorm")
    actnorm = modules.ActNorm2d(12)
    x = torch.Tensor(np.random.rand(2, 12, 64, 64))
    actnorm.initialize_parameters(x)
    y, det = actnorm(x, 0)
    x_, _ = actnorm(y, None, True)
    print("actnorm (forward,reverse) delta", float(torch.max(torch.abs(x_-x))))
    print("  det", float(det))


def test_conv1x1():
    print("[Test]: invconv1x1")
    conv = modules.InvertibleConv1x1(96)
    x = torch.Tensor(np.random.rand(2, 96, 16, 16))
    y, det = conv(x, 0)
    x_, _ = conv(y, None, True)
    print("conv1x1 (forward,reverse) delta", float(torch.max(torch.abs(x_-x))))
    print("  det", float(det))


def test_gaussian():
    # mean = torch.zeros((4, 32, 16, 16))
    # logs = torch.ones((4, 32, 16, 16))
    # x = torch.Tensor(np.random.rand(4, 32, 16, 16))
    # lh = modules.GaussianDiag.likelihood(mean, logs, x)
    # logp = modules.GaussianDiag.logp(mean, logs, x)
    pass


def test_flow_step():
    print("[Test]: flow step")
    step = models.FlowStep(32, 256, flow_coupling="affine")
    x = torch.Tensor(np.random.rand(2, 32, 16, 16))
    y, det = step(x, 0, False)
    x_, det0 = step(y, det, True)
    print("flowstep (forward,reverse)delta", float(torch.max(torch.abs(x_-x))))
    print("  det", det, det0)


def test_squeeze():
    print("[Test]: SqueezeLayer")
    layer = modules.SqueezeLayer(2)
    img = cv2.imread("pictures/tsuki.jpeg")
    img = cv2.resize(img, (256, 256))
    img = img.transpose((2, 0, 1))
    x = torch.Tensor([img])
    y, _ = layer(x, 0, False)
    x_, _ = layer(y, 0, True)
    z = y[0].numpy().transpose((1, 2, 0))
    cv2.imshow("0_3", z[:, :, 0: 3].astype(np.uint8))
    cv2.imshow("3_6", z[:, :, 3: 6].astype(np.uint8))
    cv2.imshow("6_9", z[:, :, 6: 9].astype(np.uint8))
    cv2.imshow("9_12", z[:, :, 9: 12].astype(np.uint8))
    cv2.imshow("x_", x_[0].numpy().transpose((1, 2, 0)).astype(np.uint8))
    cv2.imshow("x", x[0].numpy().transpose((1, 2, 0)).astype(np.uint8))
    cv2.waitKey()


def test_flow_net():
    print("[Test]: flow net")
    net = models.FlowNet((64, 64, 3), 256, 16, 3)
    x = torch.Tensor(np.random.rand(4, 3, 64, 64))
    y, det = net(x)
    x_ = net(y, reverse=True)
    print("z", y.size())
    print("x_", x_.size())
    print(det)


def test_glow():
    print("[Test]: Glow")
    from glow.config import JsonConfig
    glow = models.Glow(JsonConfig("hparams_celeba.json"))
    img = cv2.imread("tsuki.jpeg")
    img = cv2.resize(img, (64, 64))
    img = (img / 255.0).astype(np.float32)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    x = torch.Tensor([img]*8)
    y_onehot = torch.zeros((8, 40))
    z, det, y_logits = glow(x=x, y_onehot=y_onehot)
    print(z.size())
    print(det)
    print(models.Glow.loss_generative(det))


if __name__ == "__main__":
    test_multidim_sum()
    test_multidim_mean()
    test_actnorm()
    test_conv1x1()
    test_gaussian()
    test_flow_step()
    test_squeeze()
    test_flow_net()
    test_glow()
