# streetview
Automatically generate and complete the streetview

## DCGAN
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- Decode the simple(uniform here) distribution p_z to the images(streetview here) distribution p_data
![CITYSCAPES_DCGAN_ep0-38](/src/CITYSCAPES_DCGAN_ep0-38/CITYSCAPES_DCGAN_ep0-38.gif)
- How to use: DCGAN_streetview/ 
  `main.py --dataset datasetname --mode train`
- TODO: 
  - GAN-improved
  - VAE

**Now we can pick an arbitrary z~p_z and decode it to a image**

## Completion
- [Semantic Image Inpainting with Perceptual and Contextual Losses](https://arxiv.org/abs/1607.07539)
- Pick the z that fits the original image well
![CITYSCAPES_DCGAN_ep0-38](/src/CITYSCAPES_complete_lr/CITYSCAPES_complete_lr.gif)
- Minimizing the (Contextual + lamda*Perceptual) loss
- How to use: DCGAN_streetview/ 
  `main.py --dataset datasetname --mode complete`
- TODO: 
  - [poisson blending](http://www.ctralie.com/Teaching/PoissonImageEditing/)

**Can we encode a image to z?**

## Observations
- related to [Generative Image Modeling using Style and Structure Adversarial Networks](https://arxiv.org/abs/1603.05631)
- Here is another example which focus on pedestrian. 
- In the early training stage, the network seems to decide the *structures or poses* of pedestrians. Then in the late training stage, it only has subtle changes on *texture* according to the current batch.
![INRIA_different_batch](src/INRIA_different_batch)
- In the completion stage, it tends to choose those z resulting blurry images
![INRIA_complete_lr](src/INRIA_complete_lr)
![blurry](/src/blurry.gif)
- really make sense, but how to use it?
- TODO:
  - Inverse mapping to the latent space for GAN
  
## Some thoughts
- If we can complete the missing part of a image, can we combine one generator(pedestrian) with another discriminator(streetview)?
![ques](/src/ques.PNG)