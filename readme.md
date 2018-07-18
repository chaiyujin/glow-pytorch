# Glow
This is pytorch implementation of paper "Glow: Generative Flow with Invertible 1x1 Convolutions". Most modules are adapted from the offical TensorFlow version [openai/glow](https://github.com/openai/glow).

# TODO
- [x] Glow model. The model is coded as described in original paper, some functions are adapted from offical TF version. Most modules are tested.
- [x] Trainer, builder and hparams loaded from json.
- [ ] Test LU_decomposed 1x1 conv2d
- [ ] Infer after training

# In training
Currently, I am training the model with [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and `hparams_celeba.json`.

Following are some samples at training phase. First row is decoded by reversal flowing, second row is the original image.

![](./pictures/individualImage.png)
![](./pictures/individualImage2.png)
![](./pictures/individualImage3.png)

# Issues
There might be some errors in my codes. Please help me to figure out.
