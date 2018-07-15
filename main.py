from builder import build
from vision.datasets import CelebADataset
from train import Trainer


built = build("hparam.json", True)
dataset = CelebADataset("/home/chaiyujin/Downloads/Dataset/CelebA")

trainer = Trainer(**built, dataset=dataset, hparams="hparam.json")
trainer.train()
