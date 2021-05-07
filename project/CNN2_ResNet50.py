from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization,Activation,ZeroPadding2D,Add
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3

# caltech_dir =  '../data/image/project2/'
# categories = ["0", "1", "2", "3","4","5","6","7",
#                 "8","9"]    # y, 분류대상(파일 이름)
# nb_classes = len(categories)    # y의 갯수 (10개)

# image_w = 255   # 이미지의 너비 설정
# image_h = 255   # 이미지의 높이 설정

# pixels = image_h * image_w * 3  # shape = 255,255,3

# X = []
# y = []

# for idx, cat in enumerate(categories):
    
#     #one-hot 돌리기.
#     label = [0 for i in range(nb_classes)]
#     label[idx] = 1
#     print(label)

#     image_dir = caltech_dir + "/" + cat # caltech_dir =  '../data/image/project2/'
#     files = glob.glob(image_dir+"/*.jpg")
#     print(cat, " 파일 길이 : ", len(files))
#     # 이미지 불러오기
#     for i, f in enumerate(files):
#         img = Image.open(f)
#         img = img.convert("RGB")
#         img = img.resize((image_w, image_h))
#         data = np.asarray(img)

#         X.append(data)
#         y.append(label)

#         if i % 700 == 0:
#             print(cat, " : ", f)

# X = np.array(X)
# y = np.array(y)
# #1 0 0 0 이면 Beagle
# #0 1 0 0 이면 


X_train, X_test, y_train, y_test = train_test_split(X, y)
xy = (X_train, X_test, y_train, y_test)
np.save("../data/npy/P_project2.npy", xy)



# print("ok", len(y))

# print(X_train.shape) # (2442, 255, 255, 3)
# print(X_train.shape[0])   # 2442


X_train, X_test, y_train, y_test = np.load("../data/npy/P_project2.npy",allow_pickle=True)
print(X_train.shape)
print(X_train.shape[0])


# categories = ["Beaggle", "Bichon Frise", "Border Collie","Bulldog", "Corgi","Poodle","Retriever","Samoyed",
#                 "Schnauzer","Shih Tzu",]
# nb_classes = len(categories)

# #일반화
# X_train = X_train.astype(float) / 255
# X_test = X_test.astype(float) / 255

# idg = ImageDataGenerator(
#     width_shift_range=(0.1),   
#     height_shift_range=(0.1) 
#     ) 

# train_generator = idg.flow(X_train,y_train,batch_size=32,seed=2020)
# valid_generator = (X_test,y_test)
 
input_tensor = Input(shape=(255, 255, 3), dtype='float32', name='input')
 

def conv1_layer(x):    
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)

    return x   

    

def conv2_layer(x):         
    x = MaxPooling2D((3, 3), 2)(x)     

    shortcut = x

    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)            
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            # 안녕~~
            shortcut = x

        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            

            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  

            shortcut = x        
    
    return x



def conv3_layer(x):        
    shortcut = x    
    
    for i in range(4):     
        if(i == 0):            
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  

            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            

            x = Add()([x, shortcut])    
            x = Activation('relu')(x)    

            shortcut = x              
        
        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            

            x = Add()([x, shortcut])     
            x = Activation('relu')(x)

            shortcut = x      
            
    return x



def conv4_layer(x):
    shortcut = x        

    for i in range(6):     
        if(i == 0):            
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  

            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)

            shortcut = x               
        
        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            

            x = Add()([x, shortcut])    
            x = Activation('relu')(x)

            shortcut = x      

    return x



def conv5_layer(x):
    shortcut = x    

    for i in range(3):     
        if(i == 0):            
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            

            x = Add()([x, shortcut])  
            x = Activation('relu')(x)      

            shortcut = x               
        
        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)           
            
            x = Add()([x, shortcut]) # 중간 가중치가 엮여서 나온다.
            x = Activation('relu')(x)       

            shortcut = x                  

    return x



x = conv1_layer(input_tensor)
x = conv2_layer(x)
x = conv3_layer(x)
x = conv4_layer(x)
x = conv5_layer(x)


x = GlobalAveragePooling2D()(x)
x = Flatten() (x)

output_tensor = Dense(10, activation='softmax')(x)

# model = Model(input_tensor, output_tensor)

# model.summary()

# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5,epsilon=None), metrics=['acc'])
# model_path = '../data/modelcheckpoint/Pproject0.hdf5'
# checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
# early_stopping = EarlyStopping(monitor='val_loss', patience=150)
# # lr = ReduceLROnPlateau(patience=30, factor=0.5,verbose=1)

# learning_history = model.fit_generator(train_generator,epochs=1000, steps_per_epoch=66,
# validation_data=valid_generator, callbacks=[early_stopping,checkpoint])  