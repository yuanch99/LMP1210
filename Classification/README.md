This directory contatins the U-Net model, data and scripts.

All data used can be found in the google drive like [here](https://drive.google.com/drive/folders/1I0VnDDriFHwjLjMFvlpw74BfIGXhk-wI?usp=sharing).

The traning and cross validation data can be found in the train_val dirctory, and test data can be found in the test_dicrctory.

`unet.py` is used for transfer learning. The output weights is stored in google drive named `base_model_26th.h5`

`tuning_unet.py` is used for tuning the U-Net model after transfer learning.

`unet_resultViewer.ipynb` and `unet_usage.ipynb` is used fro plot all the U-Net figures and test data prediction.

`unet_utils.py` controls how you load the data. Please change the path to data in this file.


***Author: Yuan Chang***
