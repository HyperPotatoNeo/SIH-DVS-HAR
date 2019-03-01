import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def network(x,n_classes):
	x=TimeDistributed()(Conv2D(3,(3,3),padding='same')(x))
	x=TimeDistributed()(BatchNormalization()(x))
	x=TimeDistributed()(MaxPooling2D()(x))
	x=TimeDistributed()(ReLU()(x))
	x=TimeDistributed()(Conv2D(16,(3,3),padding='same')(x))
	x=TimeDistributed()(BatchNormalization()(x))
	x=TimeDistributed()(MaxPooling2D()(x))
	x=TimeDistributed()(ReLU()(x))
	x=TimeDistributed()(Conv2D(32,(3,3),padding='same')(x))
	x=TimeDistributed()(BatchNormalization()(x))
	x=TimeDistributed()(MaxPooling2D()(x))
	x=TimeDistributed()(ReLU()(x))
	x=TimeDistributed()(Conv2D(64,(3,3),padding='same')(x))
	x=TimeDistributed()(BatchNormalization()(x))
	x=TimeDistributed()(MaxPooling2D()(x))
	x=TimeDistributed()(ReLU()(x))
	x=TimeDistributed()(Conv2D(128,(3,3),padding='same')(x))
	x=TimeDistributed()(BatchNormalization()(x))
	x=TimeDistributed()(MaxPooling2D()(x))
	x=TimeDistributed()(ReLU()(x))
	x=TimeDistributed()(Conv2D(128,(3,3),padding='same')(x))
	x=TimeDistributed()(BatchNormalization()(x))
	x=TimeDistributed()(MaxPooling2D()(x))
	x=TimeDistributed()(ReLU()(x))
	x=TimeDistributed()(Conv2D(256,(3,3),padding='same')(x))
	x=TimeDistributed()(BatchNormalization()(x))
	x=TimeDistributed()(MaxPooling2D()(x))
	x=TimeDistributed()(ReLU()(x))
	x=Bidirectional()(LSTM(256)(x))
	x=TimeDistributed()(Dense(256)(x))
	x=TimeDistributed()(BatchNormalization()(x))
	x=TimeDistributed()(MaxPooling2D()(x))
	x=TimeDistributed()(ReLU()(x))
	output=TimeDistributed()(Dense(n_classes,activation='softmax')(x))
	return output

N_CLASSES=?

IMG_WIDTH=?
IMG_HEIGHT=?

inputs= Input(shape=(IMG_WIDTH,IMG_HEIGHT))
output= network(inputs,N_CLASSES)
model= Model(input=inputs,outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


