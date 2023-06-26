import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppressiong of information
"""
Train interface for speech enhancement!
You can just run this file.
"""
import argparse
import options
import utils
import dataloader
import datetime
import tensorflow as tf
import numpy as np
from mltk.core import summarize_model

######################################################################################################################
# Parser init
######################################################################################################################
opt = options.Options().init(argparse.ArgumentParser(description='speech enhancement')).parse_args()
print(opt)

######################################################################################################################
# Set a model (check point) and a log folder
######################################################################################################################
dir_name = os.path.dirname(os.path.abspath(__file__))  # absolute path
print(dir_name)

checkpoint_dir = os.path.join(dir_name, 'log', 'checkpoint')
saved_model_dir = os.path.join(dir_name, 'log', 'saved_model')
tensorboard_dir = os.path.join(dir_name, 'log', 'tboard_logs')
utils.mkdir(checkpoint_dir)
utils.mkdir(saved_model_dir)
utils.mkdir(tensorboard_dir)

save_name = opt.arch + '_' + opt.env
print("Now time is : ", datetime.datetime.now().isoformat())

######################################################################################################################
# Dataset
######################################################################################################################
#  Train generator
print("# Starting data loading")
generator_train = dataloader.audio_generator(opt, train_flag=True)
dataset_train = generator_train.tf_data_set
dataset_train = dataset_train.batch(opt.batch_size, drop_remainder=True).repeat()
steps_train = generator_train.num_data // opt.batch_size

# Validation generator
generator_val = dataloader.audio_generator(opt, train_flag=False)
dataset_val = generator_val.tf_data_set
dataset_val = dataset_val.batch(opt.batch_size, drop_remainder=True).repeat()
steps_val = generator_val.num_data // opt.batch_size


######################################################################################################################
# Models
######################################################################################################################

# set seeds
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'

# define model
model = utils.get_arch(opt)
model.summary()
summary = summarize_model(model)
print(summary)

# setting up callbacks
callback = []
callback.append(utils.callback_tboard(tensorboard_dir, save_name))
callback.append(utils.callback_ckpt(checkpoint_dir, save_name))

# load weights if there is pretrained model
if opt.pretrained:
    print('Load the pretrained model ...')
    model.load_weights(opt.pretrain_model_path)

######################################################################################################################
# Train
######################################################################################################################
model.fit(dataset_train,
          batch_size=None,
          epochs=opt.epochs,
          initial_epoch=opt.initial_epochs,
          verbose=1,
          steps_per_epoch=steps_train,
          validation_data=dataset_val,
          validation_steps=steps_val,
          callbacks=callback,
          max_queue_size=50,
          workers=4,
          use_multiprocessing=True
          )

model_path = os.path.join(saved_model_dir, save_name + '.h5')
model.save_weights(model_path)
tf.keras.backend.clear_session()
