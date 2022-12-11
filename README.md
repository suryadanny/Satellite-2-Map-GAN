# Satellite Image To Map Translation Using Conditional Generative Adversial Network #

The project is keras implementation of pix-2pix neural network, It implements a conditional Generative Adversarial Neural Network comprised of Deep CNN as a discriminator model 
and encoder decoder model as the Generator model.

**Research Paper used for reference -** https://arxiv.org/abs/1611.07004

**Dataset -** https://www.kaggle.com/datasets/alincijov/pix2pix-maps


**Brief Introduction-**

Genrative Neural Network Model used comprises of two components - A discriminator model , A generator model. <br>
* discriminator model is deep CNN model that's responsible to indentify that the genrated image is fake translation of satellite image or not. <br>
* generator model is encode-decoder model that takes in the satellite image and generates a map image. <br>

**How does the cGAN do its magic??** <br>

* A GAN model generates image G(z) based only on a random noise z, and since we cannot easily control the distribution of z,
it is difficult to control the output according to our desire. GAN model only ensures that G(z) is realistic, but not necessarily matches our expectation.
For instance, GAN Generator can be trained to map a random noise to a realistic representaion like a 2D Map image, the Generator will not be able to generate a image that is proper representation of the its satellite image. <br>

* cGAN solves this problem by taking both the random noise z and the input image x to produce an output image G(z|x) which looks realistic and corresponds to x. 
Since we can control which input image x is fed into the network, we can control what the output image G(z|x) . The Discriminator is trained so that D(y|x)=1 if y is BOTH realistic and correspond to x; on the other hand, D(y|x)=0 if y is either unrealistic or unrelated to input x or neither. This forces the Generator to learn to generate output G(z|x) that is both realistic and capture the overall distribution of input x and makes the image translation task possible.<br>


**Parameters -**

Satellite Images input size- 256  X  256 <br>
Learning Rate- 0.0002 <br>
Momentum (&beta;1 , &beta;2 ) -  (0.5, 0.999) <br>
&lambda; L1 - 100


**Discriminator Model - A Deep Convolution Neural Network**

* The discriminator model is a Deep convolution neural network performs simple image classificaion, takes in both the input satellite image and the 2D map image, may it be the generated or the real 2D image from the dataset.
* The output of the discriminator is based on the number of the pixels in the input image, called as a patchGAN model. model is designed such that the output prediction maps to 70 X 70 patch of the input images, the advanatage of the design being the same could be applied to larger input images.
* Model uses binary cross entropy for optimization.<br>


**Generator Model -**

* The generator is an encoder-decoder model using U-Net architecture. The moel takes in a satellite image as input image and generates a 2D Map rendering as output. The model initially downsamples the input image to bottleneck layer, after which the decoder layer decodes the bottleneck representation to the output image size.
* Skip connections are added between the encoding layer and corresponding decoding layer giving us the U-Net architecture.
* encoder and decoder layers are a combination of Convolution, Batch Normalization and LeakyRelu layers, these are extracted to common function that can reused/called while building the Generator Model.

**Training -**

* The disciminator model trains on real and the generator images. The model is initially trained in a standalone manner with real input and real output images, while the model is reused in the composite generative model.
* The generator model is train using the discriminator model, the aim of the generator model is to minimize the loss from discriminator model, that is to generate more realistic representation of the input satellite images.
* Generator is updated via weighted sum of both adversarial loss and L1 loss , 100 to 1 in favor of L1 loss, encouraging the generator to strongly generate more plausible generations of input the image.
* The gan model is composite of the generator model called before the disciminator model , the discriminator model weights that has been already trained can be used while training the generator model in the GAN.
* The composite gan model is updated with 2 targets, one defining the the generated images are real (binary cross entropy loss) , forcing large weight updates in the generator toward generating more realistic images, and the executed real translation of the image, which is compared against the output of the generator model (L1 loss).
* In the GAN model, an equilibrium is reached between the generator and discriminator model at the end, which is when our training ends.

## Results ##

Results on the validation set - 

1st Row - Satellite Image
2nd Row - Generated 2D Image
3rd Row - Real 2D Map Image



