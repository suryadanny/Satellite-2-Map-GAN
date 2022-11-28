#!/usr/bin/env python
# coding: utf-8

from os import listdir
from numpy import load

from matplotlib import pyplot
from load_utils import load_images
from numpy import savez_compressed

from utils import load_real_samples

from train_utils import train
from gan_models import define_discriminator
from gan_models import define_generator
from gan_models import define_gan


# dataset path
path = 'D:/project/archive/train/'

# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)

# save as compressed numpy array
filename = 'D:/project/archive/maps_256.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)

# load the prepared dataset
# load the dataset
data = load('D:/project/archive/maps_256.npz')
src_images, tar_images = data['arr_0'], data['arr_1']
print('Loaded: ', src_images.shape, tar_images.shape)

# plot source images
n_samples = 3
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(src_images[i].astype('uint8'))

# plot target image
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis('off')
    pyplot.imshow(tar_images[i].astype('uint8'))
pyplot.show()


# load image data
dataset = load_real_samples('D:/project/archive/maps_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]


# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)


# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)


# train model
train(d_model, g_model, gan_model, dataset)
