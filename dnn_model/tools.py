import tensorflow as tf
from pesq import pesq as get_pesq
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
import config as cfg
from keras.layers import Multiply
from pystoi import stoi as get_stoi

###############################################################################
# Calculate PESQ
###############################################################################
def cal_pesq(clean_wavs, pred_wavs):
    avg_pesq_score = 0
    for i in range(len(pred_wavs)):
        pesq_score = get_pesq(cfg.fs, clean_wavs[i], pred_wavs[i], "wb")
        avg_pesq_score += pesq_score
    avg_pesq_score /= len(pred_wavs)

    return avg_pesq_score


@tf.function
def pesq(clean_wavs, pred_wavs):
    preq_score = tf.numpy_function(cal_pesq, [clean_wavs, pred_wavs], tf.float64)

    return preq_score


###############################################################################
# Calculate STOI
###############################################################################
def cal_stoi(clean_wavs, pred_wavs):
    avg_stoi_score = 0
    for i in range(len(pred_wavs)):
        stoi_score = get_stoi(clean_wavs[i], pred_wavs[i], cfg.fs, extended=False)
        avg_stoi_score += stoi_score
    avg_stoi_score /= len(pred_wavs)

    return avg_stoi_score


@tf.function
def stoi(clean_wavs, pred_wavs):
    stoi_score = tf.numpy_function(cal_stoi, [clean_wavs, pred_wavs], tf.float64)

    return stoi_score


###############################################################################
#  Calculate Loss
###############################################################################
'''
Joint Loss (MAE + MSE)
'''

@tf.function
def loss_fn(clean_wavs, pred_wavs, r1=1, r2=1):
    r = r1 + r2
    clean_mags = tf.abs(
        tf.signal.stft(clean_wavs, frame_length=cfg.win_len, frame_step=cfg.stride, fft_length=cfg.fft_len,
                       window_fn=tf.signal.hann_window))
    pred_mags = tf.abs(
        tf.signal.stft(pred_wavs, frame_length=cfg.win_len, frame_step=cfg.stride, fft_length=cfg.fft_len,
                       window_fn=tf.signal.hann_window))

    main_loss = loss_main(clean_wavs, pred_wavs)
    sub_loss = loss_sub(clean_mags, pred_mags)
    loss = (r1 * main_loss + r2 * sub_loss) / r

    return loss


###############################################################################
#  Audio signal processing
###############################################################################
def ifft_layer(x):
    s1_stft = (tf.cast(x[0], tf.complex64) * tf.exp((1j * tf.cast(x[1], tf.complex64))))  # [None, None, 257]
    return tf.signal.inverse_stft(s1_stft, cfg.win_len, cfg.stride, fft_length=cfg.fft_len,
                                  window_fn=tf.signal.inverse_stft_window_fn(cfg.stride))

def tf_masking(x, mags):
    out = tf.squeeze(x, axis=3)
    paddings = tf.constant([[0, 0], [0, 0], [1, 0]])
    mask_mags = tf.pad(out, paddings, mode='CONSTANT')
    est_mags = Multiply()([mags, mask_mags])

    return est_mags


def spectral_mapping(x):
    out = tf.squeeze(x, axis=3)
    paddings = tf.constant([[0, 0], [0, 0], [1, 0]])
    out = tf.pad(out, paddings, mode='CONSTANT')

    return out

###############################################################################
#  Define callback
###############################################################################
tensorboard_callback = TensorBoard(log_dir=cfg.logdir + cfg.file_name[:-3])

model_checkpoint = ModelCheckpoint(filepath=cfg.chpt_dir + '/' + cfg.file_name[:-3],
                                   monitor="val_pesq",
                                   save_weights_only=True,
                                   save_best_only=True,
                                   verbose=0,
                                   mode='max',
                                   save_freq='epoch'
                                   )

# create callback for the adaptive learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode="min",
                              patience=3, verbose=1, min_lr=10 ** (-10), cooldown=1)

# create callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                               patience=10, verbose=1, mode='auto', baseline=None)

# create log file writer
csv_logger = CSVLogger(filename=cfg.logdir + cfg.file_name[:-3] + '.log')
loss_main = tf.keras.losses.MeanSquaredError()
loss_sub = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

callback = []
callback.append(tensorboard_callback)
callback.append(model_checkpoint)
# callback.append(csv_logger)
# callback.append(reduce_lr)
# callback.append(early_stopping)
