import numpy as np
import os

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics

from skimage.transform import resize
from skimage.io import imread_collection, imshow

imgs = imread_collection('imeg/*.jpeg')
print("Imported", len(imgs), "images")
print("The first one is",len(imgs[0]), "pixels tall, and",
     len(imgs[0][0]), "pixels wide")
imgs = [resize(x,(77,65),mode='constant', anti_aliasing=False) for x in imgs]
imgsarr = [x.flatten('C') for x in imgs]
plt.figure()
plt.imshow(imgs[0])
#plt.show()
#RBM
#Create a target variable: 1 through 15 for each of the 15 subjects
Y = [[_ for i in range(1,2)] for _ in range(1,3)]
Y = [num for sub in Y for num in sub]
#Define the RBM, used for feature generation
rbm = BernoulliRBM(random_state=0, verbose=True, learning_rate=.01,
                  n_iter=5, n_components=1)

#Define the Classifier - Logistic Regression will be used in this case
logistic = LogisticRegression(solver='lbfgs', max_iter=10000,
                              C=6000, multi_class='multinomial')

#Combine the two into a Pipeline
rbm_features_classifier = Pipeline(
    steps=[('rbm', rbm), ('logistic', logistic)])
# Training RBM-Logistic Pipeline
rbm_features_classifier.fit(imgsarr, Y)
Y_pred = rbm_features_classifier.predict(imgsarr)
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(Y, Y_pred)))
plt.figure(figsize=(15, 10))
plt.imshow(rbm.components_[0].reshape((77,65)), cmap=plt.cm.gray_r,
           interpolation='nearest')
plt.xticks(())
plt.yticks(())
plt.suptitle('component extracted', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()