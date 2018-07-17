import cv2
import torch
from glow.builder import build
from glow.config import JsonConfig


hparams = JsonConfig("hparams_celeba.json")
built = build(hparams, False)

graph = built["graph"]
graph.eval()
graph.z_shape = [1, 48, 8, 8]
x = graph(y_onehot=None, eps_std=0.5, reverse=True)
img = x[0].permute(1, 2, 0).detach().cpu().numpy()
img = img[:, :, ::-1]
img = cv2.resize(img, (256, 256))
cv2.imshow("img", img)
cv2.waitKey(0)
