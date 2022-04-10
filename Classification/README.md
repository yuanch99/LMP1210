This directory contatins the U-Net model, data and scripts.

All data used can be found in the google drive like [here](https://drive.google.com/drive/folders/1I0VnDDriFHwjLjMFvlpw74BfIGXhk-wI?usp=sharing) in the classification_data subdirectory. 
The traning data can be found in the train dirctory, and test data can be found in the test dicrctory.


`Unet_classification_model_train.ipynb` is used for training the decision tree classifier and logistic regression classifier models. Trained models are saved in `decision_tree.joblib` and `logistic_regression.joblib`.

`Unet_classification_model_usage.ipynb` is used for plot all the classification figures and test data predictions.

`classification_utils.py` controls how to preprocess images to input to classifiers.

Please change the data path in the `Unet_classification_model_train.ipynb` and `Unet_classification_model_usage.ipynb`.


***Author: Yuan Chang***
