# streetview
Automatically generate and complete the streetview

## DCGAN
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- Decode the simple(uniform here) distribution p_z to the images(streetview here) distribution p_data
- ![3D2G](/src/CITYSCAPES_DCGAN_3D2G/3D2G.gif)
- How to use: DCGAN_streetview/ 
  `main.py --dataset datasetname --mode train`
- TODO: 
  - GAN-improved [openai/improved-gan](https://github.com/openai/improved-gan)
  - VAE
  - step-by-step GAN
  
**Now we can pick an arbitrary z~p_z and decode it to a image**

## Completion
- [Semantic Image Inpainting with Perceptual and Contextual Losses](https://arxiv.org/abs/1607.07539)
- Pick the z that fits the original image well
- ![CITYSCAPES_complete_lr](/src/CITYSCAPES_complete_lr/CITYSCAPES_complete_lr.gif)
- Minimizing the (Contextual + lamda*Perceptual) loss
- How to use: DCGAN_streetview/ 
  `main.py --dataset datasetname --mode complete`
- TODO: 
  -  Specific generative model?

## Poisson blending
- [Poisson Image Editing](http://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)
- Synthesize new image with poisson blending
- ![str_str_poisson](/src/Poisson/14/14.gif)
- How to use: poissonblending/
  `main.py`
- TODO:
  - Heat map
  

## Observations
- related to [Generative Image Modeling using Style and Structure Adversarial Networks](https://arxiv.org/abs/1603.05631)
- Here is another example which focus on pedestrian. 
- In the early training stage, the network seems to decide the *structures or poses* of pedestrians. Then in the late training stage, it only has subtle changes on *texture* according to the current batch.
- ![INRIA_different_batch](/src/INRIA_DCGAN_2D1G/INRIA_different_batch/INRIA_different_batch.gif)
- In the completion stage, it tends to choose those z resulting blurry images
- ![blurry](/src/INRIA_DCGAN_2D1G/blurry.gif)
- TODO:
  - Inverse mapping to the latent space for GAN
  
## Some thoughts
- If we can complete the missing part of a image, can we combine one generator(pedestrian) with another discriminator(streetview)?
![ques](/src/ques.PNG)

## FCN
- [Cityscapes labels](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L26)