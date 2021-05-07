import numpy as np
import PIL
from numpy import asarray
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
import string
import scipy.signal as signal
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions


# img1=[]
# for i in range(0,6):
#     filepath='../data/image/test/%d.jfif'%i
#     image2=Image.open(filepath)
#     image2 = image2.convert('RGB')
#     image2 = image2.resize((255,255))
#     image_data2=asarray(image2)
#     # image_data2 = signal.medfilt2d(np.array(image_data2), kernel_size=3)
#     img1.append(image_data2)

 

# np.save("../data/npy/P_project_test.npy",arr=img1)
x_test = np.load("../data/npy/P_project_test.npy",allow_pickle=True)

# x = np.load("../data/npy/P_project_x3.npy",allow_pickle=True)
# y = np.load("../data/npy/P_project_y3.npy",allow_pickle=True)
# # x_test = np.load("../data/npy/P_project_test.npy",allow_pickle=True)


idg = ImageDataGenerator(
    width_shift_range=(0.1),   #
    height_shift_range=(0.1),
    zoom_range= 0.05  
    )    

x = np.load("../data/npy/P_project_x3.npy",allow_pickle=True)
y = np.load("../data/npy/P_project_y3.npy",allow_pickle=True)
# print(X_train.shape)
# print(X_train.shape[0])

print(x.shape)
print(y.shape)

import matplotlib.pyplot as plt
import cv2

plt.imshow(x_test[1], 'gray')
plt.show()

#전처리
x = x.astype(float) / 255

y = np.argmax(y, axis=1)

categories = ["Beaggle", "Bichon Frise", "Border Collie","Bulldog", "Corgi","Poodle","Retriever","Samoyed",
                "Schnauzer","Shih Tzu",]
nb_classes = len(categories)

x_train, x_valid, y_train, y_valid = train_test_split(x,y,test_size= 0.2)

train_generator = idg.flow(x_train,y_train,batch_size=8)
valid_generator = idg.flow(x_valid,y_valid)


model2 = load_model('../data/modelcheckpoint/Pproject_my.hdf5')
# model2.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001,epsilon=None), metrics=['acc'])
result = model2.evaluate(valid_generator, batch_size=16)
print(result)
# model_path = '../data/modelcheckpoint/myPproject_1.hdf5'
# checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
# early_stopping = EarlyStopping(monitor='val_loss', patience=50)
# lr = ReduceLROnPlateau(patience=25, factor=0.5,verbose=1)
# model2.summary()
# history = model2.fit_generator(train_generator,epochs=1, steps_per_epoch= len(x_train) / 8,
# validation_data=valid_generator, callbacks=[early_stopping,lr])

predict = model2.predict(x_test,verbose=True)

print(np.argmax(predict,1))