from PIL import Image
import os, glob, sys, numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils


import os, glob, numpy as np
from PIL import Image
import numpy as np
from numpy import asarray
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation
import tensorflow_hub as hub
import tensorflow as tf
import scipy.signal as signal
from keras.applications.resnet import ResNet101,preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os


# img_dir = '../data/image/project3/'
# categories = ['ect', 'dog']
# np_classes = len(categories)

# image_w = 255
# image_h = 255


# pixel = image_h * image_w * 3

# X = []
# y = []

# for idx, cat in enumerate(categories):
#     img_dir_detail = img_dir + "/" + cat
#     files = glob.glob(img_dir_detail+"/*.jpg")


#     for i, f in enumerate(files):
#         try:
#             img = Image.open(f)
#             img = img.convert("RGB")
#             img = img.resize((image_w, image_h))
#             data = np.asarray(img)
#             #Y는 0 아니면 1이니까 idx값으로 넣는다.
#             X.append(data)
#             y.append(idx)
#             if i % 300 == 0:
#                 print(cat, " : ", f)
#         except:
#             print(cat, str(i)+" 번째에서 에러 ")
# X = np.array(X)
# Y = np.array(y)


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# xy = (X_train, X_test, Y_train, Y_test)
# np.save("../data/npy/crolling.npy", xy)

img1=[]
for i in range(1,500):
    try :
        filepath='../data/image/project3/test/%d.jpg'%i
        image2=Image.open(filepath)
        image2 = image2.convert('RGB')
        image2 = image2.resize((255,255))
        image_data2=asarray(image2)
        # image_data2 = signal.medfilt2d(np.array(image_data2), kernel_size=3)
        img1.append(image_data2)
    except :
        pass
np.save("../data/npy/croll_pred.npy",arr=img1)
##########데이터 로드

x_train, x_test, y_train, y_test = np.load("../data/npy/crolling.npy",allow_pickle=True)
x_pred = np.load("../data/npy/croll_pred.npy",allow_pickle=True)
print(y_train.shape)
print(x_train.shape)

#강아지 품종분류에 사용했던 데이터셋
# x_pred = np.load("../data/npy/P_project_x4.npy",allow_pickle=True) 

#일반화
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
x_pred2 = preprocess_input(x_pred)

# resnet101 = ResNet101(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
# resnet101.trainable = False
# x = resnet101.output
# x = GlobalAveragePooling2D() (x)
# x = Flatten() (x)

# x = Dense(128) (x)
# x = BatchNormalization() (x)
# x = Activation('relu') (x)
# x = Dense(64) (x)
# x = BatchNormalization() (x)
# x = Activation('relu') (x)
# x = Dense(1, activation= 'sigmoid') (x)

# model = Model(inputs = resnet101.input, outputs = x)
# model.summary()
# model.compile(loss='binary_crossentropy', optimizer=Adam(1e-5), metrics=['acc'])


# model_path = '../data/modelcheckpoint/croll.hdf5'
# checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
# early_stopping = EarlyStopping(monitor='val_loss', patience=50)
# # lr = ReduceLROnPlateau(patience=30, factor=0.5,verbose=1)

# history = model.fit(x_train, y_train, batch_size=16, epochs=200, validation_data=(x_test, y_test),callbacks=[early_stopping,
# checkpoint])

# # print("정확도 : %.4f" % (model.evaluate(x_test, y_test)))
# print(model.evaluate(x_test, y_test))

# [0.02682698704302311, 1.0]

# import cv2
import matplotlib.pyplot as plt
#predict
model2 = load_model('../data/modelcheckpoint/croll.hdf5')


predict = model2.predict(x_pred2)
predict2 = np.round(predict, 0)
print(predict2)
for idx,per in enumerate(predict2) :
    if per == 0. :
        plt.imshow(x_pred[idx]) # 삭제 할 이미지 띄우기
        plt.show()
        print("정말로 지우시겠습니까? : y/n ")
        stc = input('') # 문자열 입력받기
        if stc == 'y' :
            print("사진을 지웠습니다.")
            os.remove('../data/image/project3/test/%d.jpg'%(idx+1))
        elif stc == 'n' :
            pass
