import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf
import uuid
from WatermarkVerification import utils

model_name = 'ACASXU_2_9'
epsilon = np.load('./data/{}.vals.npy'.format(model_name))
model = utils.load_model('./Models/{}.json'.format(model_name), './Models/{}.h5'.format(model_name))
weights = model.get_weights()
weights[-2] = weights[-2] + epsilon

model.set_weights(weights)
model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

utils.save_model('./Models/{}_corrected.json'.format(model_name), './Models/{}_corrected.h5'.format(model_name), model)
utils.saveModelAsProtobuf(model, '{}_corrected'.format(model_name))
