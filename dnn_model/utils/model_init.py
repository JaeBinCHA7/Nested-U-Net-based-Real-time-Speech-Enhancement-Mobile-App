import tensorflow as tf
from pesq import pesq as get_pesq
from pystoi import stoi as get_stoi
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
import os


###############################################################################
#                                 Model Init                                  #
###############################################################################

# get architecture
def get_arch(opt):
    arch = opt.arch
    optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr_initial)

    print('You choose ' + arch + '...')
    if arch == 'NUTLS-LSTM':
        from models import NUTLS_LSTM

        nutls_lstm = NUTLS_LSTM(opt)
        model = nutls_lstm.build_model()
        model.compile(optimizer=optimizer, loss=loss_func, metrics=[pesq, stoi], run_eagerly=False)

    else:
        raise Exception("Arch error!")

    return model


###############################################################################
#                              Calculate Loss                                 #
###############################################################################
@tf.function
def loss_func(clean_wavs, pred_wavs, r1=1, r2=1, win_len=512, fft_len=512, hop_len=256):
    loss_mse = tf.keras.losses.MeanSquaredError()
    loss_mae = tf.keras.losses.MeanAbsoluteError()

    r = r1 + r2
    clean_mags = tf.abs(
        tf.signal.stft(clean_wavs, frame_length=win_len, frame_step=hop_len, fft_length=fft_len,
                       window_fn=tf.signal.hann_window))
    pred_mags = tf.abs(
        tf.signal.stft(pred_wavs, frame_length=win_len, frame_step=hop_len, fft_length=fft_len,
                       window_fn=tf.signal.hann_window))

    main_loss = loss_mse(clean_wavs, pred_wavs)
    sub_loss = loss_mae(clean_mags, pred_mags)
    loss = (r1 * main_loss + r2 * sub_loss) / r

    return loss


###############################################################################
#                              Calculate PESQ                                 #
###############################################################################
def cal_pesq(clean_wavs, pred_wavs, fs=16000):
    avg_pesq_score = 0.0
    num = len(pred_wavs)
    for i in range(num):
        pesq_score = 0.0
        try:
            pesq_score = get_pesq(fs, clean_wavs[i], pred_wavs[i], "wb")
        except:
            num -= 1
        avg_pesq_score += pesq_score
    avg_pesq_score /= num

    return avg_pesq_score


@tf.function
def pesq(clean_wavs, pred_wavs):
    pesq = tf.numpy_function(cal_pesq, [clean_wavs, pred_wavs], tf.float64)

    return pesq


###############################################################################
#                              Calculate STOI                                 #
###############################################################################
def cal_stoi(clean_wavs, pred_wavs, fs=16000):
    avg_stoi_score = 0
    for i in range(len(pred_wavs)):
        stoi_score = get_stoi(clean_wavs[i], pred_wavs[i], fs, extended=False)
        avg_stoi_score += stoi_score
    avg_stoi_score /= len(pred_wavs)

    return avg_stoi_score


@tf.function
def stoi(clean_wavs, pred_wavs):
    stoi_score = tf.numpy_function(cal_stoi, [clean_wavs, pred_wavs], tf.float64)

    return stoi_score


###############################################################################
#                           Setting up callbacks                              #
###############################################################################
def callback_tboard(path, name):
    tensorboard_callback = TensorBoard(log_dir=os.path.join(path, name))

    return tensorboard_callback


def callback_ckpt(path, name):
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(path, name),
                                       monitor="val_pesq",
                                       save_weights_only=True,
                                       save_best_only=True,
                                       verbose=0,
                                       mode='max',
                                       save_freq='epoch'
                                       )
    return model_checkpoint


def callback_reduce_lr():
    # create callback for the adaptive learning rate
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode="min",
                                  patience=3, verbose=1, min_lr=10 ** (-10), cooldown=1)

    return reduce_lr


def callback_stop_early():
    # create callback for early stopping
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                   patience=10, verbose=1, mode='auto', baseline=None)

    return early_stopping
