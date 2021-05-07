import os
import os, glob, numpy as np
from PIL import Image
import numpy as np
from numpy import asarray
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
import tensorflow as tf
import scipy.signal as signal
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
#########데이터 로드

caltech_dir =  '../../data/image/train/'
categories = []
for i in range(0,1001) :
    i = "%d"%i
    categories.append(i)

nb_classes = len(categories)

image_w = 255
image_h = 255

pixels = image_h * image_w * 3

X = []
y = []

for idx, cat in enumerate(categories):
    
    #one-hot 돌리기.
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    print(cat, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        y.append(label)

        if i % 700 == 0:
            print(cat, " : ", f)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)
xy = (X_train, X_test, y_train, y_test)
np.save("../data/npy/P_project.npy", xy)
# x_pred = np.load("../data/npy/P_project_test.npy",allow_pickle=True)
x_train, x_test, y_train, y_test = np.load("../data/npy/P_project.npy",allow_pickle=True)


# categories = categories
# nb_classes = len(categories)


#일반화
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)


resent = ResNet50(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
resent.trainable = False
x = resent.output
# resnet101 = ResNet101(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
# resnet101.trainable = False
# x = resnet101.output
# x = Conv2D(1026, kernel_size=(3,3), strides=(1,1), padding='valid') (x)
# x = Conv2D(1026, kernel_size=(3,3), strides=(1,1), padding='valid',) (x)
# x = BatchNormalization() (x)
# x = Activation('relu') (x)
x = GlobalAveragePooling2D() (x)
x = Flatten() (x)

x = Dense(128) (x)
x = BatchNormalization() (x)
x = Activation('relu') (x)
x = Dense(64) (x)
x = BatchNormalization() (x)
x = Activation('relu') (x)
x = Dense(1000, activation= 'softmax') (x)

model = Model(inputs = resent.input, outputs = x)


# resent.trainable = True

# for layer_ in model.layers :
#     print(layer_)
#     print(layer_.get_output_at(0).get_shape().as_list())


#false로 적용 했을 때 결과값이 엉망.... true는 적용안했을때와 결과와 속도가 같았다.
# summary결과 또한 적용 안한것과 파라미터가 같은것으로 보아 true가 디폴트인듯 하다


#위 결과와 똑같았다. 둘의 차이점은 없는듯하다. 다만 for문은 resent.layer[:100]과 같은 방법으로 세세하게 조정이 가능한듯 하다.

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['acc'])


model_path = '../../data/modelcheckpoint/lotte.hdf5'
checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
lr = ReduceLROnPlateau(patience=5, factor=0.5,verbose=1)

history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test),
callbacks=[early_stopping,checkpoint,lr])
print("정확도 : %.4f" % (model.evaluate(x_test, y_test)[1]))




# model2 = load_model('../data/modelcheckpoint/Pproject_fine.hdf5')
# predict = model2.predict(x_pred)
# # print("정확도 : %.4f" % (model2.evaluate(x_test, y_test)[1]))
# # print(np.argmax(predict,1))
# # false 정확도 : 0.8596
# # True 정확도 : 0.9261
# categories = ["Beaggle", "Bichon Frise", "Border Collie","Bulldog", "Corgi","Poodle","Retriever","Samoyed",
#                 "Schnauzer","Shih Tzu",]
# for i in np.argmax(predict,1) :
#     if i ==0 :
#         print("이 사진은",categories[0],"입니다")
#     if i ==1 :
#         print("이 사진은",categories[1],"입니다")
#     if i ==2 :
#         print("이 사진은",categories[2],"입니다")
#     if i ==3 :
#         print("이 사진은",categories[3],"입니다")
#     if i ==4 :
#         print("이 사진은",categories[4],"입니다")
#     if i ==5 :
#         print("이 사진은",categories[5],"입니다")
#     if i ==6 :
#         print("이 사진은",categories[6],"입니다")
#     if i ==7 :
#         print("이 사진은",categories[7],"입니다")
#     if i ==8 :
#         print("이 사진은",categories[8],"입니다")   
#     if i ==9 :
#         print("이 사진은",categories[9],"입니다")


