from glow.builder import build
from vision.datasets import CelebADataset
from torchvision import transforms
from glow.trainer import Trainer
from glow.config import JsonConfig


hparams = JsonConfig("hparams_celeba.json")
built = build(hparams, True)
dataset = CelebADataset("/Users/chaiyujin/Downloads/database/CelebA",
                        transforms.Compose([
                                           transforms.CenterCrop(hparams.Data.center_crop),
                                           transforms.Resize(hparams.Data.resize),
                                           transforms.ToTensor()]))

trainer = Trainer(**built, dataset=dataset, hparams=hparams)
trainer.train()
