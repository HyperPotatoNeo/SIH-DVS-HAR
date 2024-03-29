import numpy as np
from argparse import ArgumentParser
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
from tensorflow.keras.layers import Activation, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from dataloader import *

def get_parser():
	parser = ArgumentParser()
	parser.add_argument('-e', '--epochs', dest = 'epochs', default = 50)
	parser.add_argument('-i', '--initial-epoch', dest = 'initial-epoch', default = 0)
	args = vars(parser.parse_args())

	return args

args = get_parser()

N_CLASSES=5
N_FRAMES=5
IMG_WIDTH=240
IMG_HEIGHT=180

generator = VideoGenerator(
	train_dir = 'data/train',
	test_dir = 'data/test',
	dims = (N_FRAMES, IMG_WIDTH, IMG_HEIGHT, N_CLASSES),
	batch_size = 1,
	shuffle = True,
	file_ext = '.npy'
	)

train_gen = generator.generate(
	mode = 'train'
	)

test_gen = generator.generate(
	mode = 'test'
	)

checkpoint = ModelCheckpoint(
	'models/weights_{epoch:02d}_{val_acc:.2f}.hdf5',
	monitor = 'val_acc',
	verbose = 1,
	save_best_only = True,
	mode = 'max'
	)

tensorboard = TensorBoard(
	log_dir = './logs',
	histogram_freq = 1,
	write_graph = True,
	update_freq  ='epoch'
	)

# inputs = Input(shape = (N_FRAMES,IMG_WIDTH,IMG_HEIGHT))
# x=TimeDistributed(Conv2D(3,(3,3),padding='same'))(inputs)
# x=TimeDistributed(BatchNormalization())(x)
# x=TimeDistributed(MaxPooling2D())(x)
# x=TimeDistributed(ReLU())(x)
# x=TimeDistributed(Conv2D(16,(3,3),padding='same'))(x)
# x=TimeDistributed(BatchNormalization())(x)
# x=TimeDistributed(MaxPooling2D())(x)
# x=TimeDistributed(ReLU())(x)
# x=TimeDistributed(Conv2D(32,(3,3),padding='same'))(x)
# x=TimeDistributed(BatchNormalization())(x)
# x=TimeDistributed(MaxPooling2D())(x)
# x=TimeDistributed(ReLU())(x)
# x=TimeDistributed(Conv2D(64,(3,3),padding='same'))(x)
# x=TimeDistributed(BatchNormalization())(x)
# x=TimeDistributed(MaxPooling2D())(x)
# x=TimeDistributed(ReLU())(x)
# x=TimeDistributed(Conv2D(128,(3,3),padding='same'))(x)
# x=TimeDistributed(BatchNormalization())(x)
# x=TimeDistributed(MaxPooling2D())(x)
# x=TimeDistributed(ReLU())(x)
# x=TimeDistributed(Conv2D(128,(3,3),padding='same'))(x)
# x=TimeDistributed(BatchNormalization())(x)
# x=TimeDistributed(MaxPooling2D())(x)
# x=TimeDistributed(ReLU())(x)
# x=TimeDistributed(Conv2D(256,(3,3),padding='same'))(x)
# x=TimeDistributed(BatchNormalization())(x)
# x=TimeDistributed(MaxPooling2D())(x)
# x=TimeDistributed(ReLU())(x)
# x=Bidirectional(LSTM(256))(x)
# x=TimeDistributed()(Dense(256)(x))
# x=TimeDistributed(BatchNormalization())(x)
# x=TimeDistributed(MaxPooling2D())(x)
# x=TimeDistributed(ReLU())(x)
# output=TimeDistributed(Dense(n_classes, activation='softmax'))(x)
# output = network(inputs,N_CLASSES)
# model = Model(input=inputs,outputs=output)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model=Sequential()
model.add(TimeDistributed(Conv2D(3,(3,3),padding='same'),input_shape=(N_FRAMES,IMG_WIDTH,IMG_HEIGHT,1)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D()))
model.add(TimeDistributed(Conv2D(32,(3,3),padding='same')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D()))
model.add(TimeDistributed(Conv2D(64,(3,3),padding='same')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D()))
model.add(TimeDistributed(Conv2D(128,(3,3),padding='same')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D()))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(256))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(5,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
	generator = train_gen,
	steps_per_epoch = 742,
	epochs = args['epochs'],
	verbose = 1,
	initial_epoch = args['initial-epoch'],
	callbacks = [checkpoint]#, tensorboard]
	#validation_data = test_gen,
	#validation_steps = ?
	)