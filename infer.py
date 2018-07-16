import torch
from builder import build
from vision.datasets import CelebADataset, MNISTDataset
from torchvision import transforms
from train import Trainer
from config import JsonConfig
import cv2


hparams = JsonConfig("hparams.1.json")
built = build(hparams, False)
dataset = CelebADataset("/home/chaiyujin/Downloads/Dataset/CelebA",
                        transforms.Compose([
                                           transforms.CenterCrop(hparams.Data.center_crop),
                                           transforms.Resize(hparams.Data.resize),
                                           transforms.ToTensor()]))
d = dataset[1]
y_onehot = torch.Tensor([d["y_onehot"]]).cuda()
print(y_onehot)

y_onehot = torch.Tensor([[0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,
          1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  1.]]).cuda()

y_onehot = torch.Tensor([[ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
          0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,
          1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,
          1.,  0.,  0.,  1.]]).cuda()
graph = built["graph"]
graph.eval()
graph.z_shape = [1, 48, 8, 8]
x = graph(y_onehot=y_onehot, eps_std=0.5, reverse=True)
img = x[0].permute(1, 2, 0).detach().cpu().numpy()
img = img[:, :, ::-1]
img = cv2.resize(img, (256, 256))
cv2.imshow("img", img)
cv2.waitKey(0)
