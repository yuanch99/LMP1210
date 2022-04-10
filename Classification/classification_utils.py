# import image_slicer
import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_image(path):
  im = cv2.imread(path)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  n_h=(im.shape[0]//411)
  n_w=(im.shape[1]//411)
  im=im[0:n_h*411, 0:n_w*411]
  k=0
  for r in range(0,im.shape[0],411):
    for c in range(0,im.shape[1],411):
        if k == 0 :
          imgs=cv2.resize(im[r:r+411, c:c+411,:], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
          imgs=np.expand_dims(imgs,axis=0)
        else:
          subim=cv2.resize(im[r:r+411, c:c+411,:], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
          imgs=np.concatenate((imgs,np.expand_dims(subim,axis=0)))
        k+=1
  return  imgs

