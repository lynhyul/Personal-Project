
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization,Activation,ZeroPadding2D,Add
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam

# from PIL import Image
# import os, glob, numpy as np

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
# #1 0 0 0 이면 Beagle
# #0 1 0 0 이면 

# print(X.shape)
# print(y.shape)


# X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle = False)
# xy = (X_train, X_test, y_train, y_test)
# np.save("../data/npy/P_project_xy2.npy", arr=xy)
# # np.save("../data/npy/P_project_y.npy", arr=y)



# print("ok", len(y))

# print(X_train.shape) # (2442, 255, 255, 3)
# print(X_train.shape[0])   # 2442


# X_train, X_test, y_train, y_test = np.load("../data/npy/P_project.npy",allow_pickle=True)
# # x = np.load("../data/npy/P_project_x.npy",allow_pickle=True)
# # y = np.load("../data/npy/P_project_y.npy",allow_pickle=True)
# # print(X_train.shape)
# # print(X_train.shape[0])

# print(x.shape)
# print(y.shape)
x = np.load("../data/npy/P_project_x3.npy",allow_pickle=True)
y = np.load("../data/npy/P_project_y3.npy",allow_pickle=True)

categories = ["Beaggle", "Bichon Frise", "Border Collie","Bulldog", "Corgi","Poodle","Retriever","Samoyed",
                "Schnauzer","Shih Tzu",]
nb_classes = len(categories)

#일반화
# X_train = X_train.astype(float) / 255
# X_test = X_test.astype(float) / 255
x = x.astype(float) / 255

y = np.argmax(y, axis=1)    # kfold로 적용시키기 위해 1차원의 형태로 변환
                            # 이는 나중에 sparse_categorical_generator를 이용

# print(X_train.shape)    # 2433, 255,255,3
# print(X_test.shape)     # 812, 255, 255, 3

idg = ImageDataGenerator(
    width_shift_range=(0.1),   
    height_shift_range=(0.1) 
    ) 

# train_generator = idg.flow(x_train,y_train,batch_size=32,seed=2020)
# valid_generator = idg.flow(x_test,y_test)
 
input_tensor = Input(shape=(255, 255, 3), dtype='float32', name='input')
 
from sklearn.model_selection import StratifiedKFold, KFold

skf = StratifiedKFold(n_splits=8, random_state=42, shuffle=True)

   
    
acc1 = []
nth = 0

for train_index, valid_index in skf.split(x,y) :  

    x_train = x[train_index]
    x_valid = x[valid_index]    
    y_train = y[train_index]
    y_valid = y[valid_index]
    print(x_train.shape)

    train_generator = idg.flow(x_train,y_train,batch_size=32,seed=2020)
    valid_generator = idg.flow(x_valid,y_valid)

    def conv1_layer(resent):    
        resent = ZeroPadding2D(padding=(3, 3))(resent)
        resent = Conv2D(64, (7, 7), strides=(2, 2))(resent)
        resent = BatchNormalization()(resent)
        resent = Activation('relu')(resent)
        resent = ZeroPadding2D(padding=(1,1))(resent)
    
        return resent   
    
        
    
    def conv2_layer(resent):         
        resent = MaxPooling2D((3, 3), 2)(resent)     
    
        shortcut = resent
    
        for i in range(3):
            if (i == 0):
                resent = Conv2D(16, (1, 1), strides=(1, 1), padding='valid')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)
                
                resent = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)
    
                resent = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(resent)
                shortcut = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(shortcut)            
                resent = BatchNormalization()(resent)
                shortcut = BatchNormalization()(shortcut)
    
                resent = Add()([resent, shortcut])
                resent = Activation('relu')(resent)
                
                shortcut = resent
    
            else:
                resent = Conv2D(16, (1, 1), strides=(1, 1), padding='valid')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)
                
                resent = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)
    
                resent = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(resent)
                resent = BatchNormalization()(resent)            
    
                resent = Add()([resent, shortcut])   
                resent = Activation('relu')(resent)  
    
                shortcut = resent        
        
        return resent
    
    
    
    def conv3_layer(resent):        
        shortcut = resent    
        
        for i in range(4):     
            if(i == 0):            
                resent = Conv2D(32, (1, 1), strides=(2, 2), padding='valid')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)        
                
                resent = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)  
    
                resent = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(resent)
                shortcut = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(shortcut)
                resent = BatchNormalization()(resent)
                shortcut = BatchNormalization()(shortcut)            
    
                resent = Add()([resent, shortcut])    
                resent = Activation('relu')(resent)    
    
                shortcut = resent              
            
            else:
                resent = Conv2D(32, (1, 1), strides=(1, 1), padding='valid')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)
                
                resent = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)
    
                resent = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(resent)
                resent = BatchNormalization()(resent)            
    
                resent = Add()([resent, shortcut])     
                resent = Activation('relu')(resent)
    
                shortcut = resent      
                
        return resent
    
    
    
    def conv4_layer(resent):
        shortcut = resent        
    
        for i in range(2):     
            if(i == 0):            
                resent = Conv2D(64, (1, 1), strides=(2, 2), padding='valid')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)        
                
                resent = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)  
    
                resent = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(resent)
                shortcut = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(shortcut)
                resent = BatchNormalization()(resent)
                shortcut = BatchNormalization()(shortcut)
    
                resent = Add()([resent, shortcut]) 
                resent = Activation('relu')(resent)
    
                shortcut = resent               
            
            else:
                resent = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)
                
                resent = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)
    
                resent = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(resent)
                resent = BatchNormalization()(resent)            
    
                resent = Add()([resent, shortcut])    
                resent = Activation('relu')(resent)
    
                shortcut = resent      
    
        return resent
    

    
    def conv5_layer(resent):
        shortcut = resent    
    
        for i in range(2):     
            if(i == 0):            
                resent = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)        
                
                resent = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)  
    
                resent = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(resent)
                shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
                resent = BatchNormalization()(resent)
                shortcut = BatchNormalization()(shortcut)            
    
                resent = Add()([resent, shortcut])  
                resent = Activation('relu')(resent)      
    
                shortcut = resent               
            
            else:
                resent = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)
                
                resent = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(resent)
                resent = BatchNormalization()(resent)
                resent = Activation('relu')(resent)
    
                resent = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(resent)
                resent = BatchNormalization()(resent)           
                
                resent = Add()([resent, shortcut]) # 중간 가중치가 엮여서 나온다.
                resent = Activation('relu')(resent)       
    
                shortcut = resent                  
    
        return resent
    
    
    
    resent = conv1_layer(input_tensor)
    resent = conv2_layer(resent)
    # resent = conv3_layer(resent)
    # resent = conv4_layer(resent)
    # resent = conv5_layer(resent)
    
    
    resent = GlobalAveragePooling2D()(resent)
    # resent = Activation('softmax')(resent)
    # resent = MaxPooling2D(pool_size=(2,2)) (resent)
    resent = Dropout(0.5) (resent)
    resent = Flatten() (resent)

    resent = Dense(128, activation= 'relu') (resent)
    resent = BatchNormalization() (resent)
    resent = Dense(64, activation= 'relu') (resent)
    resent = BatchNormalization() (resent)
    resent = Dropout(0.2) (resent)

    output_tensor = Dense(10, activation='softmax')(resent)
    
    model = Model(input_tensor, output_tensor)

    # model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001,epsilon=None), 
    metrics=['sparse_categorical_accuracy'])
    model_path = '../data/modelcheckpoint/Pproject_my.hdf5'
    checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=150)
    lr = ReduceLROnPlateau(patience=30, factor=0.5,verbose=1)

    history = model.fit_generator(train_generator,
    epochs=500, validation_data=valid_generator, callbacks=[early_stopping,checkpoint])
    acc = model.evaluate(valid_generator)
    # print(acc) # [2.324734926223755, 0.10474430024623871]
    acc1.append(acc[1])
    nth += 1
    print(nth, '번째 학습을 완료했습니다.')  
print("정확도 : %.4f" % (acc1[1]))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])       #회귀모델이기때문에 acc측정이 힘들다
plt.plot(history.history['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['tran loss', 'val loss', 'train acc','val acc'])    #주석
plt.show()
