# Efficient net B7



import os
import os, glob, numpy as np
from PIL import Image
import numpy as np
from numpy import asarray
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
import tensorflow_hub as hub
import tensorflow as tf
import scipy.signal as signal
from tensorflow.keras.applications import VGG16, VGG19
from keras.applications.vgg19 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

##########데이터 로드
x_train, x_test, y_train, y_test = np.load("../data/npy/P_project.npy",allow_pickle=True)


categories = ["Beaggle", "Bichon Frise", "Border Collie","Bulldog", "Corgi","Poodle","Retriever","Samoyed",
                "Schnauzer","Shih Tzu",]
nb_classes = len(categories)


#일반화
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)


VGG19 = VGG19(include_top=True,weights='imagenet',input_shape=(224,224,3))
# ,input_shape=x_train.shape[1:]
# VGG19.trainable = False
# x = VGG19.output
# # x = MaxPooling2D(pool_size=(2,2)) (x)
# x = Dropout(0.5) (x)
# x = Flatten() (x)

# x = Dense(128, activation= 'relu') (x)
# x = BatchNormalization() (x)
# x = Dense(64, activation= 'relu') (x)
# x = BatchNormalization() (x)
# x = Dropout(0.2) (x)
# x = Dense(10, activation= 'softmax') (x)

VGG19.summary()

# model = Model(inputs = VGG19.input, outputs = x)
# model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['acc'])

# model.summary()

# model_path = '../data/modelcheckpoint/Pproject9.hdf5'
# checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
# early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# # lr = ReduceLROnPlateau(patience=30, factor=0.5,verbose=1)

# history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test),callbacks=[early_stopping,
# checkpoint])

# print("정확도 : %.4f" % (model.evaluate(x_test, y_test)[1]))
# print(model.evaluate(x_test, y_test))

# import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.plot(history.history['acc'])       #회귀모델이기때문에 acc측정이 힘들다
# plt.plot(history.history['val_acc'])
# plt.title('loss & acc')
# plt.ylabel('loss & acc')
# plt.xlabel('epoch')
# plt.legend(['tran loss', 'val loss', 'train acc','val acc'])    #주석
# plt.show()

# # F정확도 : 0.8737
# # T정확도 : 0.9021
# #[0.4091291129589081, 0.8793103694915771]