# streetview
Automatically generate and complete the streetview

## DCGAN
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- Decode the simple(uniform here) distribution p_z to the images(streetview here) distribution p_data
- Use: DCGAN_streetview/ main.py --dataset datasetname --mode train
- ![CITYSCAPES_DCGAN_ep0-38](/src/CITYSCAPES_DCGAN_ep0-38/CITYSCAPES_DCGAN_ep0-38.gif)
- TODO: GAN-improved, VAE

Now we can pick an arbitrary z~p_z and decode it to a image

## Completion
- [Semantic Image Inpainting with Perceptual and Contextual Losses](https://arxiv.org/abs/1607.07539)
- Pick the z that fits the original image well
- Minimizing the (Contextual + lamda*Perceptual) loss
- Use: DCGAN_streetview/ main.py --dataset datasetname --mode complete
- TODO: [poisson blending](http://www.ctralie.com/Teaching/PoissonImageEditing/)

But what's behind this? Can we encode a image to z?

## 
