import os, fnmatch
import numpy as np
import soundfile as sf
import config as cfg
import tensorflow as tf
from random import shuffle


# Data Normalization
def minMaxNorm(wav, eps=1e-8):
    max = np.max(abs(wav))
    min = np.min(abs(wav))
    wav = (wav - min) / (max - min + eps)
    return wav


class audio_generator():
    def __init__(self, path_to_noisy, path_to_clean, train_flag=False):
        self.path_to_clean = path_to_clean
        self.path_to_noisy = path_to_noisy
        self.noisy_files_list = fnmatch.filter(os.listdir(path_to_noisy), '*.wav')
        self.num_data = len(self.noisy_files_list)
        self.train_flag = train_flag
        self.create_tf_data_obj()
        self.chuck_size = cfg.fs * 3

    def create_generator(self):
        if self.train_flag:
            shuffle(self.noisy_files_list)
        for noisy_file in self.noisy_files_list:
            clean_file = noisy_file[:8]  # WSJ0
            clean_file += '.wav'

            noisy_speech, fs = sf.read(os.path.join(self.path_to_noisy, noisy_file))
            clean_speech, fs = sf.read(os.path.join(self.path_to_clean, clean_file))

            # Select speech of chuck size
            if len(noisy_speech) > self.chuck_size:
                start_point = np.random.randint(0, len(noisy_speech) - self.chuck_size)
                noisy_speech = noisy_speech[start_point:start_point + self.chuck_size]
                clean_speech = clean_speech[start_point:start_point + self.chuck_size]
            elif len(noisy_speech) < self.chuck_size:
                noisy_speech = np.concatenate([noisy_speech, noisy_speech[:self.chuck_size - len(noisy_speech)]],
                                              axis=0)
                clean_speech = np.concatenate([clean_speech, clean_speech[:self.chuck_size - len(clean_speech)]],
                                              axis=0)

                if len(noisy_speech) < self.chuck_size:
                    noisy_speech = np.concatenate([noisy_speech, noisy_speech[:self.chuck_size - len(noisy_speech)]],
                                                  axis=0)
                    clean_speech = np.concatenate([clean_speech, clean_speech[:self.chuck_size - len(clean_speech)]],
                                                  axis=0)

            noisy_speech, clean_speech = minMaxNorm(noisy_speech), minMaxNorm(clean_speech)

            noisy_speech = tf.convert_to_tensor(noisy_speech)
            clean_speech = tf.convert_to_tensor(clean_speech)

            noisy_speech = tf.clip_by_value(noisy_speech, -1, 1)
            clean_speech = tf.clip_by_value(clean_speech, -1, 1)

            yield noisy_speech, clean_speech

    def create_tf_data_obj(self):
        # creating the tf.data.Dataset from the iterator
        self.tf_data_set = tf.data.Dataset.from_generator(
            self.create_generator,
            (tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([self.chuck_size]), tf.TensorShape([self.chuck_size])),
            args=None
        )


#################################################################################
#  Train generator
#################################################################################
generator_train = audio_generator(cfg.path_to_train_noisy,
                                  cfg.path_to_train_clean,
                                  train_flag=True,
                                  )
dataset_train = generator_train.tf_data_set
dataset_train = dataset_train.batch(cfg.BATCH_SIZE, drop_remainder=True).repeat()
steps_train = generator_train.num_data // cfg.BATCH_SIZE

#################################################################################
#  Validation generator
#################################################################################
generator_val = audio_generator(cfg.path_to_val_noisy,
                                cfg.path_to_val_clean,
                                train_flag=True
                                )
dataset_val = generator_val.tf_data_set
dataset_val = dataset_val.batch(cfg.BATCH_SIZE, drop_remainder=True).repeat()
steps_val = generator_val.num_data // cfg.BATCH_SIZE
