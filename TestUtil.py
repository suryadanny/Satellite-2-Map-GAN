#!/usr/bin/env python
# coding: utf-8


from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
from numpy import expand_dims
from matplotlib import pyplot
from keras.models import load_model


def load_test_image(file_path, size=(256, 256)):
    img = load_img(file_path, target_size=size)
    img_arr = img_to_array(img)
    print(img_arr.shape)
    pyplot.imshow(img_arr)
    img_arr = (img_arr - 127.5) / 127.5
    pyplot.imshow(img_arr)
    img_arr = expand_dims(img_arr, 0)
    pyplot.imshow(img_arr[0])
    return img_arr


filePath = 'D:/project/'
imgArr = load_test_image(filePath + 'test14.jpg')

pyplot.imshow(imgArr[0])

model = load_model('D:/project/sat-gan/Satellite-2-Map-GAN/model_034200.h5')
map_img = model.predict(imgArr)
map_img = (map_img + 1) / 2

pyplot.imshow(map_img[0])
