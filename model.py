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
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from dataloader import *

def get_parser():
	parser = ArgumentParser()
	parser.add_argument('-e', '--epochs', dest = 'epochs', default = 40)
	parser.add_argument('-i', '--initial_epoch', dest = 'epochs', default = 0)
	

def network(x, n_classes):
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
	output=TimeDistributed()(Dense(n_classes, activation='softmax')(x))
	return output

N_CLASSES=?
N_FRAMES = ?
IMG_WIDTH=?
IMG_HEIGHT=?

generator = VideoGenerator(
	train_dir = 'data/train',
	test_Dir = 'data/test',
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
	mode = 'max'
	)

tensorboard = TensorBoard(
	log_dir = './logs',
	histogram_freq = 1,
    write_graph = True,
    update_freq  ='epoch'
    )

inputs = Input(shape = (N_FRAMES,IMG_WIDTH,IMG_HEIGHT))
output = network(inputs,N_CLASSES)
model = Model(input=inputs,outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
	generator = train_gen,
	steps_per_epoch = ?,
	epochs = ?,
	verbose = 1,
	callbacks = [checkpoint, tensorboard],
	validation_data = test_gen,
	validation_steps = ?
	)