from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model,load_model,save_model
from keras.layers import Input, Activation
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization,Activation,ZeroPadding2D,Add
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3

# caltech_dir =  '../data/image/project/'
# categories = ["0", "1", "2", "3","4","5","6","7",
#                 "8","9"] 
# nb_classes = len(categories)

# image_w = 255
# image_h = 255

# pixels = image_h * image_w * 3

# X = []
# y = []

# for idx, cat in enumerate(categories):
    
#     #one-hot 돌리기.
#     label = [0 for i in range(nb_classes)]
#     label[idx] = 1

#     image_dir = caltech_dir + "/" + cat
#     files = glob.glob(image_dir+"/*.jpg")
#     print(cat, " 파일 길이 : ", len(files))
#     for i, f in enumerate(files):
#         img = Image.open(f)
#         img = img.convert("RGB")
#         img = img.resize((image_w, image_h))
#         data = np.asarray(img)

#         X.append(data)
#         y.append(label)


# X = np.array(X)
# y = np.array(y)
# # #1 0 0 0 이면 Beagle
# # #0 1 0 0 이면 

# # print(X.shape)
# # print(y.shape)


# # # X_train, X_test, y_train, y_test = train_test_split(X, y)
# # # xy = (X_train, X_test, y_train, y_test)
# np.save("../data/npy/P_project_x4.npy", arr=X)
# np.save("../data/npy/P_project_y4.npy", arr=y)



# print("ok", len(y))

# print(X_train.shape) # (2442, 255, 255, 3)
# print(X_train.shape[0])   # 2442


# X_train, X_test, y_train, y_test = np.load("../data/npy/P_project.npy",allow_pickle=True)
x = np.load("../data/npy/P_project_x4.npy",allow_pickle=True)
y = np.load("../data/npy/P_project_y4.npy",allow_pickle=True)
# x_test = np.load("../data/npy/P_project_test.npy",allow_pickle=True)


idg = ImageDataGenerator(
    width_shift_range=(0.1),   #
    height_shift_range=(0.1),
    zoom_range= 0.05  
    )    


x = np.load("../data/npy/P_project_x4.npy",allow_pickle=True)
y = np.load("../data/npy/P_project_y4.npy",allow_pickle=True)
# print(X_train.shape)
# print(X_train.shape[0])

print(x.shape)
print(y.shape)

x_test = np.load("../data/npy/P_project_test.npy",allow_pickle=True)
#전처리
x = x.astype(float) / 255
x_test = x_test.astype(float) / 255

y = np.argmax(y, axis=1)

categories = ["Beaggle", "Bichon Frise", "Border Collie","Bulldog", "Corgi","Poodle","Retriever","Samoyed",
                "Schnauzer","Shih Tzu",]
nb_classes = len(categories)

x_train, x_valid, y_train, y_valid = train_test_split(x,y,test_size= 0.2)

# from sklearn.model_selection import StratifiedKFold, KFold
# skf = StratifiedKFold(n_splits=8, random_state=42, shuffle=True)

# nth = 0

# for train_index, valid_index in skf.split(x,y) :  

#     x_train = x[train_index]
#     x_valid = x[valid_index]    
#     y_train = y[train_index]
#     y_valid = y[valid_index]

train_generator = idg.flow(x_train,y_train,batch_size=8)
valid_generator = idg.flow(x_valid,y_valid)

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size =(3,3), padding = 'valid', strides=(1,1),
input_shape = (255,255,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))                   
model.add(Conv2D(filters = 32, kernel_size =(3,3), padding = 'valid',strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters = 64, kernel_size =(3,3), padding = 'valid',strides=(1,1)))
model.add(BatchNormalization())    
model.add(Activation('relu'))
model.add(Dropout(0.2))                         
model.add(Conv2D(filters = 64, kernel_size =(3,3), padding = 'valid',strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters = 128, kernel_size =(3,3), padding = 'valid', strides=(1,1)))
model.add(BatchNormalization())     
model.add(Activation('relu'))
model.add(Dropout(0.2))                        
model.add(Conv2D(filters = 128, kernel_size =(3,3), padding = 'valid', strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(Conv2D(filters = 128, kernel_size =(3,3), padding = 'valid', strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

for i in range (2) :
    model.add(Conv2D(filters = 256, kernel_size =(3,3), padding = 'valid', strides=(1,1)))
    model.add(BatchNormalization())     
    model.add(Activation('relu'))
    model.add(Dropout(0.2))                        
    model.add(Conv2D(filters = 256, kernel_size =(3,3), padding = 'valid' ,strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2)) 
    model.add(Conv2D(filters = 256, kernel_size =(3,3), padding = 'valid' ,strides=(1,1)))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
    model.add(Dropout(0.2)) 
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))
# model2 = load_model('../data/modelcheckpoint/myPproject_5.hdf5')
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001,epsilon=None), metrics=['acc'])

model.summary()
model_path = '../data/modelcheckpoint/my_project_act2.hdf5'
checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_acc', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=50)
lr = ReduceLROnPlateau(patience=25, factor=0.5,verbose=1)

history = model.fit_generator(train_generator,epochs=200, steps_per_epoch= len(x_train) / 8,
validation_data=valid_generator, callbacks=[early_stopping,lr,checkpoint])

print(model.evaluate(valid_generator)[1])
# nth = nth +1
# print(nth,"번째 완료")


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])      
plt.plot(history.history['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['tran loss', 'val loss', 'train acc','val acc'])    #주석
plt.show()  
model2 = load_model('../data/modelcheckpoint/my_project_act2.hdf5')
# model2.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001,epsilon=None), metrics=['acc'])

result = model2.predict_generator(x_test,verbose=True)
print(np.argmax(result,1))

categories = ["Beaggle", "Bichon Frise", "Border Collie","Bulldog", "Corgi","Poodle","Retriever","Samoyed",
            "Schnauzer","Shih Tzu",]
for i in np.argmax(result,1) :
    if i ==0 :
        print("이 사진은",categories[0],"입니다")
    if i ==1 :
        print("이 사진은",categories[1],"입니다")
    if i ==2 :
        print("이 사진은",categories[2],"입니다")
    if i ==3 :
        print("이 사진은",categories[3],"입니다")
    if i ==4 :
        print("이 사진은",categories[4],"입니다")
    if i ==5 :
        print("이 사진은",categories[5],"입니다")
    if i ==6 :
        print("이 사진은",categories[6],"입니다")
    if i ==7 :
        print("이 사진은",categories[7],"입니다")
    if i ==8 :
        print("이 사진은",categories[8],"입니다")   
    if i ==9 :
        print("이 사진은",categories[9],"입니다")

