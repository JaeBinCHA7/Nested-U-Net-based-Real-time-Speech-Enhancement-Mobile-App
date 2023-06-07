import os, fnmatch
import numpy as np
import soundfile as sf
import tensorflow as tf
from random import shuffle
import math
import re


# Data Normalization
def minMaxNorm(wav, eps=1e-8):
    max = np.max(abs(wav))
    min = np.min(abs(wav))
    wav = (wav - min) / (max - min + eps)
    return wav


class audio_generator():
    def __init__(self, opt, train_flag=False, indexing_flag=False):
        self.train_flag = train_flag
        if self.train_flag:
            self.path_to_clean = opt.clean_dirs_for_train
            self.path_to_noisy = opt.noisy_dirs_for_train
        else:
            self.path_to_clean = opt.clean_dirs_for_valid
            self.path_to_noisy = opt.clean_dirs_for_valid

        self.noisy_files_list = fnmatch.filter(os.listdir(self.path_to_noisy), '*.wav')
        self.hop_len = opt.hop_len

        if opt.chunk_size % opt.hop_len != 0:
            self.chuck_size = opt.chunk_size + (self.hop_len // 2)
        else:
            self.chuck_size = opt.chunk_size

        self.num_data = len(self.noisy_files_list)
        self.create_tf_data_obj()

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


class testset_geneator():
    def __init__(self, opt):
        self.path_to_clean = opt.clean_dirs_for_test
        self.path_to_noisy = opt.noisy_dirs_for_test
        self.noisy_files_list = fnmatch.filter(os.listdir(self.path_to_noisy), '*.wav')
        self.num_data = len(self.noisy_files_list)
        self.create_tf_data_obj()

    def create_generator(self):
        for noisy_file in self.noisy_files_list:
            clean_file = noisy_file[:8]  # WSJ0
            clean_file += '.wav'

            noisy_speech, fs = sf.read(os.path.join(self.path_to_noisy, noisy_file))
            clean_speech, fs = sf.read(os.path.join(self.path_to_clean, clean_file))

            noisy_speech, clean_speech = minMaxNorm(noisy_speech), minMaxNorm(clean_speech)

            if (len(noisy_speech) != len(clean_speech)):
                print("The length of noisy data and clean data does not match.")

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
            output_shapes=(tf.TensorShape([None]), tf.TensorShape([None])),
        )


class mapping_testset():
    def __init__(self, opt):
        self.path_to_clean = opt.clean_dirs_for_test
        self.path_to_noisy = opt.noisy_dirs_for_test
        self.noisy_files_list = fnmatch.filter(os.listdir(opt.noisy_dirs_for_test), '*.wav')
        self.opt = opt

    def mapping_data(self):
        clean_list = []
        noisy_list = []

        for noisy_file in self.noisy_files_list:
            clean_file = noisy_file[:8]  # WSJ0
            clean_file += '.wav'

            noisy_speech, fs = sf.read(os.path.join(self.path_to_noisy, noisy_file))
            clean_speech, fs = sf.read(os.path.join(self.path_to_clean, clean_file))

            noisy_speech, clean_speech = minMaxNorm(noisy_speech), minMaxNorm(clean_speech)

            noisy_speech = tf.convert_to_tensor(noisy_speech)
            clean_speech = tf.convert_to_tensor(clean_speech)

            noisy_speech = tf.clip_by_value(noisy_speech, -1, 1)
            clean_speech = tf.clip_by_value(clean_speech, -1, 1)

            if len(noisy_speech) % self.opt.hop_len != 0:
                noisy_speech = np.pad(noisy_speech, [self.opt.hop_len - (len(noisy_speech) % self.opt.hop_len), 0])
                clean_speech = np.pad(clean_speech, [self.opt.hop_len - (len(clean_speech) % self.opt.hop_len), 0])

            noisy_speech = np.expand_dims(noisy_speech, axis=0)
            # clean_speech = np.expand_dims(clean_speech, axis=0)

            noisy_speech = np.array(noisy_speech)
            clean_speech = np.array(clean_speech)

            clean_list.append(clean_speech)
            noisy_list.append(noisy_speech)

        return clean_list, noisy_list

    def get_snr_index(self):
        snr_index = []
        for i, noisy in enumerate(self.noisy_files_list):
            snr = ''.join(re.findall(r'\d+', noisy[8:]))
            snr_index.append(snr)

        return snr_index
