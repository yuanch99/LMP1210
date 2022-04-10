#!/usr/bin/env python
# coding: utf-8

# In[1]:


import traceback
import segmentation_models as sm
from tensorflow import keras
import loadData
#from tf_processe_data import display_sample
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd

# In[ ]:


sm.set_framework('tf.keras')
sm.framework()
#define model
BACKBONE = 'vgg19'
BATCH_SIZE = 8
TRAIN_SIZE = 141
VAL_SIZE = 10
CLASSES = ['NCI', 'GCI','_background_']
LR = 0.00001

preprocess_input = sm.get_preprocessing(BACKBONE)

model = sm.Unet(BACKBONE, encoder_weights=None,classes=3, activation='softmax',input_shape=(None, None, 3),encoder_freeze=False)
model.load_weights('base_model_26th.h5')
optim = keras.optimizers.Adam(LR)
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
loss_funcs = sm.losses.CategoricalCELoss(class_weights=np.array([1, 1, 1]))
#loss_funcs = sm.losses.CategoricalCELoss
model.compile(optim, loss_funcs, metrics)

model.summary()


# In[ ]:


x_train,y_train = loadData.load_data()
x_train = preprocess_input(x_train)
y_train = y_train.astype('float32')
y_train = y_train/2

x_test = loadData.load_test()
x_test = preprocess_input(x_test)


# In[ ]:


#fitting parameters
callbacks = [
    keras.callbacks.ModelCheckpoint('./tuning_model_000_new_base.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]


EPOCHS = 50

STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
VALIDATION_STEPS = VAL_SIZE // BATCH_SIZE


# In[ ]:


model_history = model.fit(x_train,y_train, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_split=0.1,
                              callbacks=callbacks)


# In[ ]:


#save tuning history
pd.DataFrame(model_history.history).to_csv('tuning_model_000_newbase.csv')


# In[ ]:


#save prediction
i=134
pr_mask = model.predict(np.expand_dims(x_train[i],axis=0))
loadData.display_sample([x_train[i], y_train[i],np.squeeze(pr_mask, axis=0)],'/hpf/largeprojects/tabori/users/yuan/lmp1210/data/output/tuning_model.png')


# In[ ]:


#plot the masks
pred=np.squeeze(pr_mask, axis=0)
plt.imshow(pred[...,0], interpolation='nearest')


# In[ ]:




