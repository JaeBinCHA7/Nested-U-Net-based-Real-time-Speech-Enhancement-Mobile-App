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
from pesq import pesq as get_pesq
from pystoi import stoi as get_stoi

######################################################################################################################
# Parser init
######################################################################################################################
opt = options.Options().init(argparse.ArgumentParser(description='Real-time speech enhancement')).parse_args()
print(opt)

######################################################################################################################
# Dataset
######################################################################################################################
generator_test = dataloader.testset_geneator(opt)
dataset_test = generator_test.tf_data_set
dataset_test = dataset_test.batch(opt.test_batch, drop_remainder=True)
steps_test = generator_test.num_data // opt.test_batch

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

model.load_weights(opt.pretrain_model_path)

######################################################################################################################
# Test
######################################################################################################################
# Calculate performance by SNR
mt = dataloader.mapping_testset(opt)
clean_list, noisy_list = mt.mapping_data()
snr_index = mt.get_snr_index()

pesq_list = []
stoi_list = []
for i, noisy_wav in enumerate(noisy_list):
    pred_wav = model(noisy_wav, training=False)
    pred_wav = np.array(pred_wav)
    pesq_score = get_pesq(opt.fs, clean_list[i], pred_wav, "wb")
    stoi_score = get_stoi(clean_list[i], pred_wav, opt.fs, extended=False)
    pesq_list.append(pesq_score)
    stoi_list.append(stoi_score)

avg_pesq_0, avg_pesq_5, avg_pesq_10, avg_pesq_15 = 0, 0, 0, 0
avg_stoi_0, avg_stoi_5, avg_stoi_10, avg_stoi_15 = 0, 0, 0, 0

for i, snr in enumerate(snr_index):
    num_0, num_5, num_10, num_15 = 0, 0, 0, 0
    if snr == '0':
        avg_pesq_0 += pesq_list[i]
        avg_stoi_0 += stoi_list[i]
        num_0 += 1
    elif snr == '5':
        avg_pesq_5 += pesq_list[i]
        avg_stoi_5 += stoi_list[i]
        num_5 += 1
    elif snr == '10':
        avg_pesq_10 += pesq_list[i]
        avg_stoi_10 += stoi_list[i]
        num_10 += 1
    elif snr == '15':
        avg_pesq_15 += pesq_list[i]
        avg_stoi_15 += stoi_list[i]
        num_15 += 1

print("###########################################")
print("# {}".format(model.name))
print("###########################################")
print("# Testset performance [0dB]")
print('# PESQ : {}'.format(avg_pesq_0 / num_0))
print('# STOI : {}'.format(avg_stoi_0 / num_0))
print("###########################################")
print("# Testset performance [5dB]")
print('# PESQ : {}'.format(avg_pesq_5 / num_5))
print('# STOI : {}'.format(avg_stoi_5 / num_5))
print("###########################################")
print("# Testset performance [10dB]")
print('# PESQ : {}'.format(avg_pesq_10 / num_10))
print('# STOI : {}'.format(avg_stoi_10 / num_10))
print("###########################################")
print("# Testset performance [15dB]")
print('# PESQ : {}'.format(avg_pesq_15 / num_15))
print('# STOI : {}'.format(avg_stoi_15 / num_15))
print("###########################################")
print(" Average of PESQ : {}".format(
    (avg_pesq_0 + avg_pesq_5 + avg_pesq_10 + avg_pesq_15) / (num_0 + num_5 + num_10 + num_15)))
print(" Average of STOI : {}".format(
    (avg_stoi_0 + avg_stoi_5 + avg_stoi_10 + avg_stoi_15) / (num_0 + num_5 + num_10 + num_15)))
print("###########################################")
