import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import cv2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops


# config = tf.compat.v1.ConfigProto() 
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split = 0.2
)

'''
- rotation_range: 이미지 회전 범위 (degrees)
- width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 
                                (원본 가로, 세로 길이에 대한 비율 값)
- rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 
            모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 
            그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 
            이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
- shear_range: 임의 전단 변환 (shearing transformation) 범위
- zoom_range: 임의 확대/축소 범위
- horizontal_flip`: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 
    원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
- fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
'''

xy_train = train_datagen.flow_from_directory(
    '../data/image/project/',        
    target_size = (255,255),
    batch_size= 4,  
    class_mode='categorical', 
    subset = 'training'

)


xy_test = train_datagen.flow_from_directory(
    '../data/image/project/',       
    target_size = (255,255),
    batch_size= 4,
    class_mode='categorical', 
    subset = 'validation'
)
print(xy_train[0][1].shape)
# data_flow = generator.flow(data.x_train, data.y_train, batch_size=batch_size)
# 로 이미지를 생성해서 반환하는 일종의 Iterator(반복자)를 만듭니다.
# 이 반복자는 x, y를 가지고 한번에 batch_size만큼의 랜덤하게 변형된 학습 데이터를 만들어줍니다.


# with tf_ops.device('/device:GPU:0'):

#     model = Sequential()

#     model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape =(255,255,3), activation= 'relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))
    
#     model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))
    
#     model.add(Flatten())
#     model.add(Dense(256, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(10, activation='softmax'))
#     model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics=['acc'])

#     from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#     es = EarlyStopping(monitor= 'val_loss', patience=30)
#     lr = ReduceLROnPlateau(monitor='val_loss', patience=15, factor=0.5)

    history = model.fit_generator(xy_train, epochs=500, validation_data=xy_test)
#     callbacks=[es,lr])
# # steps_per_epoch=32 => 32개에 대한 데이터를 1에포에 대해서 32번만 학습?

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss'] 
# val_loss = history.history['val_loss']


# print('acc : ', acc[-1])
# # print('loss : ', loss[:-1])
# # print('val_acc : ', val_loss[:-1])


