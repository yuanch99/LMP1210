#pip install segmentation-models
import traceback
import segmentation_models as sm
from tensorflow import keras
import tf_processe_data
from tf_processe_data import parse_image,display_sample,load_image_train,load_image_test
import tensorflow as tf
import numpy as np
import loadData

sm.set_framework('tf.keras')
sm.framework()
#define model
BACKBONE = 'vgg19'
BATCH_SIZE = 8
TRAIN_SIZE = 141
VAL_SIZE = 10
CLASSES = ['GCI', 'NCI']
LR = 0.001
EPOCHS = 40

preprocess_input = sm.get_preprocessing(BACKBONE)

model = sm.Unet(BACKBONE, encoder_weights='imagenet',classes=3, activation='softmax',input_shape=(None, None, 3),encoder_freeze=True)
#model.summary()

optim = keras.optimizers.Adam(LR)
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
loss_funcs = sm.losses.CategoricalFocalLoss() + sm.losses.DiceLoss(class_weights=np.array([1, 1.5,1])) 
model.compile(optim, loss_funcs, metrics)

#load data

img_path = '/Users/yuanchang/Documents/MBP/ML/project/data/train_val/figure/'


BATCH_SIZE = 16
# BUFFER_SIZE = 500

x_train,y_train = loadData.load_data()
x_train = preprocess_input(x_train)
y_train = y_train.astype('float32')
y_train = y_train/2

x_test = loadData.load_test()
x_test = preprocess_input(x_test)

# for image, mask in dataset['train'].take(1):
#     sample_image, sample_mask = image, mask

# for i in range(16):
#   display_sample([sample_image[i], sample_mask[i]])


#fit
callbacks = [
    keras.callbacks.ModelCheckpoint('./base_model_26th.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

# x_train = preprocess_input(dataset['train'])
# x_val = preprocess_input(dataset['val'])

STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
VALIDATION_STEPS = VAL_SIZE // BATCH_SIZE

# On CPU
with tf.device("/cpu:0"):
    model_history = model.fit(x_train,y_train, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_split=0.1,
                              callbacks=callbacks)

