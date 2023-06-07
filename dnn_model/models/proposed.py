'''
model for real-time speech enhancement
baseline(Nested U-Net TLS) + lstm unit /2
'''

import keras
from keras import Model, Input
import tensorflow as tf
from keras.layers import *
import numpy as np


class NUTLS_LSTM():
    def __init__(self, opt):
        self.in_ch = 1
        self.mid_ch = 32
        self.out_ch = 64
        self.win_len = opt.win_len
        self.fft_len = opt.fft_len
        self.hop_len = opt.hop_len
        self.unit = 21
        self.opt = opt
        ############################################################################
        # Train
        ############################################################################
        self.msfe6_en_lstm = LSTM(self.unit, return_sequences=True, name="msfe6_en_lstm")
        self.msfe6_en_dense = Dense(128, name="msfe6_en_dense")

        self.msfe5_en_lstm = LSTM(self.unit, return_sequences=True, name="msfe5_en_lstm")
        self.msfe5_en_dense = Dense(128, name="msfe5_en_dense")

        self.msfe4_en_lstm_1 = LSTM(self.unit, return_sequences=True, name="msfe4_en_lstm")
        self.msfe4_en_dense_1 = Dense(128, name="msfe4_en_dense")

        self.msfe4_en_lstm_2 = LSTM(self.unit, return_sequences=True, name="msfe4_en2_lstm")
        self.msfe4_en_dense_2 = Dense(64, name="msfe4_en2_dense")

        self.msfe4_en_lstm_3 = LSTM(self.unit, return_sequences=True, name="msfe4_en3_lstm")
        self.msfe4_en_dense_3 = Dense(32, name="msfe4_en3_dense")

        self.msfe3_en_lstm = LSTM(self.unit, return_sequences=True, name="msfe3_en_lstm")
        self.msfe3_en_dense = Dense(32, name="msfe3_en_dense")

        self.lstm = LSTM(self.unit, return_sequences=True, name="lstm")
        self.dense = Dense(256, name="dense")

        self.msfe3_de_lstm = LSTM(self.unit, return_sequences=True, name="msfe3_de_lstm")
        self.msfe3_de_dense = Dense(32, name="msfe6_de_dense")

        self.msfe4_de_lstm_1 = LSTM(self.unit, return_sequences=True, name="msfe4_de_lstm")
        self.msfe4_de_dense_1 = Dense(32, name="msfe5_de_dense")

        self.msfe4_de_lstm_2 = LSTM(self.unit, return_sequences=True, name="msfe4_de2_lstm")
        self.msfe4_de_dense_2 = Dense(64, name="msfe4_de_dense")

        self.msfe4_de_lstm_3 = LSTM(self.unit, return_sequences=True, name="msfe4_de3_lstm")
        self.msfe4_de_dense_3 = Dense(128, name="msfe4_de2_dense")

        self.msfe5_de_lstm = LSTM(self.unit, return_sequences=True, name="msfe5_de_lstm")
        self.msfe5_de_dense = Dense(128, name="msfe4_de3_dense")

        self.msfe6_de_lstm = LSTM(self.unit, return_sequences=True, name="msfe6_de_lstm")
        self.msfe6_de_dense = Dense(128, name="msfe3_de_dense")

        self.out_conv = Conv2D(self.in_ch, kernel_size=1)

        ############################################################################
        # TFLite
        ############################################################################
        self.msfe6_en_lstm_tfl = LSTM(self.unit, return_sequences=True, unroll=True, return_state=True,
                                      name="msfe6_en_lstm")
        self.msfe6_en_dense_tfl = Dense(128, name="msfe6_en_dense")

        self.msfe5_en_lstm_tfl = LSTM(self.unit, return_sequences=True, unroll=True, return_state=True,
                                      name="msfe5_en_lstm")
        self.msfe5_en_dense_tfl = Dense(128, name="msfe5_en_dense")

        self.msfe4_en_lstm_1_tfl = LSTM(self.unit, return_sequences=True, unroll=True, return_state=True,
                                        name="msfe4_en_lstm")
        self.msfe4_en_dense_1_tfl = Dense(128, name="msfe4_en_dense")

        self.msfe4_en_lstm_2_tfl = LSTM(self.unit, return_sequences=True, unroll=True, return_state=True,
                                        name="msfe4_en2_lstm")
        self.msfe4_en_dense_2_tfl = Dense(64, name="msfe4_en2_dense")

        self.msfe4_en_lstm_3_tfl = LSTM(self.unit, return_sequences=True, unroll=True, return_state=True,
                                        name="msfe4_en3_lstm")
        self.msfe4_en_dense_3_tfl = Dense(32, name="msfe4_en3_dense")

        self.msfe3_en_lstm_tfl = LSTM(self.unit, return_sequences=True, unroll=True, return_state=True,
                                      name="msfe3_en_lstm")
        self.msfe3_en_dense_tfl = Dense(32, name="msfe3_en_dense")

        self.lstm_tfl = LSTM(self.unit, return_sequences=True, unroll=True, return_state=True, name="lstm")
        self.dense_tfl = Dense(256, name="dense")

        self.msfe3_de_lstm_tfl = LSTM(self.unit, return_sequences=True, unroll=True, return_state=True,
                                      name="msfe3_de_lstm")
        self.msfe3_de_dense_tfl = Dense(32, name="msfe3_de_dense")

        self.msfe4_de_lstm_1_tfl = LSTM(self.unit, return_sequences=True, unroll=True, return_state=True,
                                        name="msfe4_de_lstm")
        self.msfe4_de_dense_1_tfl = Dense(32, name="msfe4_de_dense")

        self.msfe4_de_lstm_2_tfl = LSTM(self.unit, return_sequences=True, unroll=True, return_state=True,
                                        name="msfe4_de2_lstm")
        self.msfe4_de_dense_2_tfl = Dense(64, name="msfe4_de2_dense")

        self.msfe4_de_lstm_3_tfl = LSTM(self.unit, return_sequences=True, unroll=True, return_state=True,
                                        name="msfe4_de3_lstm")
        self.msfe4_de_dense_3_tfl = Dense(128, name="msfe4_de3_dense")

        self.msfe5_de_lstm_tfl = LSTM(self.unit, return_sequences=True, unroll=True, return_state=True,
                                      name="msfe5_de_lstm")
        self.msfe5_de_dense_tfl = Dense(128, name="msfe5_de_dense")

        self.msfe6_de_lstm_tfl = LSTM(self.unit, return_sequences=True, unroll=True, return_state=True,
                                      name="msfe6_de_lstm")
        self.msfe6_de_dense_tfl = Dense(128, name="msfe6_de_dense")

    ############################################################################
    # Tools
    ############################################################################
    # CTFA (Causal Time Frequency Attention)
    def ctfa(self, x, in_ch, out_ch=16, time_seq=32, name_ta=None, name_fa=None):
        B, T, F, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        if B == None:
            B = self.opt.batch_size
        time_attention = keras.Sequential([
            Permute((2, 1, 3)),
            Reshape([F, T * C]),
            GlobalAveragePooling1D(),
            Reshape([T, C]),
            Conv1D(out_ch, kernel_size=1),
            ReLU(),
            Conv1D(in_ch, kernel_size=1),
            Activation('sigmoid'),
            Reshape([T, 1, C]),
            Lambda(lambda x: tf.broadcast_to(x, shape=[B, T, F, C]))

        ], name=name_ta)

        freq_attention = keras.Sequential([
            ZeroPadding2D(padding=((time_seq - 1, 0), (0, 0))),
            Reshape([T + time_seq - 1, F * C]),
            AveragePooling1D(time_seq, strides=1),
            Reshape([T, F, C]),
            Conv2D(out_ch, kernel_size=1),
            ReLU(),
            Conv2D(in_ch, kernel_size=1),
            Activation('sigmoid')
        ], name=name_fa)

        TA = time_attention(x)
        FA = freq_attention(TA)

        TFA = FA * TA
        out = x * TFA

        return out

    def ctfa_rt(self, x, in_ch, out_ch=16, time_seq=32, name_ta=None, name_fa=None):
        T, F, C = x.shape[1], x.shape[2], x.shape[3]

        time_attention = keras.Sequential([
            Permute((2, 1, 3)),
            Reshape([F, T * C]),
            GlobalAveragePooling1D(),
            Reshape([T, C]),
            Conv1D(out_ch, kernel_size=1),
            ReLU(),
            Conv1D(in_ch, kernel_size=1),
            Activation('sigmoid'),
            Reshape([T, 1, C]),
            Lambda(lambda x: tf.broadcast_to(x, shape=[1, T, F, C]))

        ], name=name_ta)

        freq_attention = keras.Sequential([
            ZeroPadding2D(padding=((time_seq - 1, 0), (0, 0))),
            Reshape([T + time_seq - 1, F * C]),
            AveragePooling1D(time_seq, strides=1),
            Reshape([T, F, C]),
            Conv2D(out_ch, kernel_size=1),
            ReLU(),
            Conv2D(in_ch, kernel_size=1),
            Activation('sigmoid')
        ], name=name_fa)

        TA = time_attention(x)
        FA = freq_attention(TA)

        TFA = FA * TA
        out = x * TFA

        return out

    def conv(self, out_ch, x, name=None):
        conv2d = keras.Sequential([
            ZeroPadding2D(padding=((1, 0), (1, 1))),
            Conv2D(out_ch, kernel_size=(2, 3), strides=(1, 2), padding='valid'),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return conv2d(x)

    def conv_valid(self, out_ch, x, name=None):
        conv2d = keras.Sequential([
            ZeroPadding2D(padding=((0, 0), (1, 1))),
            Conv2D(out_ch, kernel_size=(2, 3), strides=(1, 2), padding='valid'),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return conv2d(x)

    def inconv(self, out_ch, x, name=None):
        in_conv = keras.Sequential([
            Conv2D(out_ch, kernel_size=1),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return in_conv(x)

    def spconv(self, out_ch, x, scale_factor=2, name=None):
        sp_conv = keras.Sequential([
            ZeroPadding2D(padding=((1, 0), (1, 1))),
            Conv2D(out_ch * scale_factor, kernel_size=(2, 3), strides=1, padding='valid'),  # [B, T, F, C]
            Reshape((-1, x.shape[2], x.shape[3] // scale_factor, scale_factor)),  # [B, T, F, C//2 , 2],
            Permute((1, 2, 4, 3)),  # [B, T, F, 2, C//2]
            Reshape((-1, x.shape[2] * scale_factor, out_ch)),  # [B, T, 2*F, C//2]
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return sp_conv(x)

    def spconv_valid(self, out_ch, x, scale_factor=2, name=None):
        sp_conv = keras.Sequential([
            ZeroPadding2D(padding=((0, 0), (1, 1))),
            Conv2D(out_ch * scale_factor, kernel_size=(2, 3), strides=1, padding='valid'),  # [B, T, F, C]
            Reshape((-1, x.shape[2], x.shape[3] // scale_factor, scale_factor)),  # [B, T, F, C//2 , 2],
            Permute((1, 2, 4, 3)),  # [B, T, F, 2, C//2]
            Reshape((-1, x.shape[2] * scale_factor, out_ch)),  # [B, T, 2*F, C//2]
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return sp_conv(x)

    def down_sampling(self, in_ch, x, name=None):
        ds = keras.Sequential([
            Conv2D(in_ch, kernel_size=(1, 3), strides=(1, 2), padding='same')
        ], name=name)

        return ds(x)

    def up_sampling(self, in_ch, x, name=None):
        us = keras.Sequential([
            Conv2DTranspose(in_ch, kernel_size=(1, 3), strides=(1, 2), padding='same')
        ], name=name)

        return us(x)

    # Reshape (Before LSTM)
    def reshape_before_lstm(self, x):
        shape = np.shape(x)
        if shape[1] == None:
            reshape_out = Reshape((-1, shape[2] * shape[3]))(x)
        else:
            reshape_out = Reshape((shape[1], shape[2] * shape[3]))(x)
        return reshape_out, shape

    # Reshape (After LSTM)
    def reshape_after_lstm(self, x, shape):
        if shape[1] == None:
            lstm_out = Reshape((-1, shape[2], shape[3]))(x)
        else:
            lstm_out = Reshape((shape[1], shape[2], shape[3]))(x)
        return lstm_out

    def train_model(self, x):
        frames = tf.signal.stft(x, frame_length=self.win_len, frame_step=self.hop_len, fft_length=self.fft_len,
                                window_fn=tf.signal.hann_window)
        mags = tf.abs(frames)  # [None, None, 257]
        phase = tf.math.angle(frames)  # [None, None, 257]

        trans_mags = tf.expand_dims(mags, axis=3)
        trans_mags = trans_mags[:, :, 1:, :]

        en_in = self.inconv(self.out_ch, trans_mags, name="input_layer")
        ############################################################################
        # MSFE6 - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, en_in, name="msfe6_en_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe6_en_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe6_en_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe6_en_conv3")
        en4_out = self.conv(self.mid_ch, en3_out, name="msfe6_en_conv4")
        en5_out = self.conv(self.mid_ch, en4_out, name="msfe6_en_conv5")
        en6_out = self.conv(self.mid_ch, en5_out, name="msfe6_en_conv6")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en6_out)
        en_lstm = self.msfe6_en_lstm(reshape_out)
        en_lstm = self.msfe6_en_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de6_out6 = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en6_out]), name="msfe6_en_spconv1")
        de6_out5 = self.spconv(self.mid_ch, Concatenate(axis=3)([de6_out6, en5_out]), name="msfe6_en_spconv2")
        de6_out4 = self.spconv(self.mid_ch, Concatenate(axis=3)([de6_out5, en4_out]), name="msfe6_en_spconv3")
        de6_out3 = self.spconv(self.mid_ch, Concatenate(axis=3)([de6_out4, en3_out]), name="msfe6_en_spconv4")
        de6_out2 = self.spconv(self.mid_ch, Concatenate(axis=3)([de6_out3, en2_out]), name="msfe6_en_spconv5")
        de6_out1 = self.spconv(self.out_ch, Concatenate(axis=3)([de6_out2, en1_out]), name="msfe6_en_spconv6")

        de_out = self.ctfa(de6_out1, self.out_ch, name_ta='msfe6_en_ta', name_fa='msfe6_en_fa') + en_in

        # Down-sampling
        msfe6_out = self.down_sampling(self.out_ch, de_out, name="msfe6_down_sampling")

        ############################################################################
        # MSFE5 - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, msfe6_out, name="msfe5_en_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe5_en_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe5_en_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe5_en_conv3")
        en4_out = self.conv(self.mid_ch, en3_out, name="msfe5_en_conv4")
        en5_out = self.conv(self.mid_ch, en4_out, name="msfe5_en_conv5")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en5_out)
        en_lstm = self.msfe5_en_lstm(reshape_out)
        en_lstm = self.msfe5_en_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de5_out5 = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en5_out]), name="msfe5_en_spconv1")
        de5_out4 = self.spconv(self.mid_ch, Concatenate(axis=3)([de5_out5, en4_out]), name="msfe5_en_spconv2")
        de5_out3 = self.spconv(self.mid_ch, Concatenate(axis=3)([de5_out4, en3_out]), name="msfe5_en_spconv3")
        de5_out2 = self.spconv(self.mid_ch, Concatenate(axis=3)([de5_out3, en2_out]), name="msfe5_en_spconv4")
        de5_out1 = self.spconv(self.out_ch, Concatenate(axis=3)([de5_out2, en1_out]), name="msfe5_en_spconv5")

        de_out = self.ctfa(de5_out1, self.out_ch, name_ta='msfe5_en_ta', name_fa='msfe5_en_fa') + en_in

        # Down-sampling
        msfe5_out = self.down_sampling(self.out_ch, de_out, name="msfe5_down_sampling")  # [None, 372, 128 ,64]

        ############################################################################
        # MSFE4 - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, msfe5_out, name="msfe4_en_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe4_en_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe4_en_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe4_en_conv3")
        en4_out = self.conv(self.mid_ch, en3_out, name="msfe4_en_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        en_lstm = self.msfe4_en_lstm_1(reshape_out)
        en_lstm = self.msfe4_en_dense_1(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de4_out4 = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en4_out]), name="msfe4_en_spconv1")
        de4_out3 = self.spconv(self.mid_ch, Concatenate(axis=3)([de4_out4, en3_out]), name="msfe4_en_spconv2")
        de4_out2 = self.spconv(self.mid_ch, Concatenate(axis=3)([de4_out3, en2_out]), name="msfe4_en_spconv3")
        de4_out1 = self.spconv(self.out_ch, Concatenate(axis=3)([de4_out2, en1_out]), name="msfe4_en_spconv4")

        de_out = self.ctfa(de4_out1, self.out_ch, name_ta='msfe4_en_ta', name_fa='msfe4_en_fa') + en_in

        # Down-sampling
        msfe4_out = self.down_sampling(self.out_ch, de_out, name="msfe4_down_sampling")

        ############################################################################
        # MSFE4(2) - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, msfe4_out, name="msfe4_en2_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe4_en2_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe4_en2_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe4_en2_conv3")
        en4_out = self.conv(self.mid_ch, en3_out, name="msfe4_en2_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        en_lstm = self.msfe4_en_lstm_2(reshape_out)
        en_lstm = self.msfe4_en_dense_2(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de4_2_out4 = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en4_out]), name="msfe4_en2_spconv1")
        de4_2_out3 = self.spconv(self.mid_ch, Concatenate(axis=3)([de4_2_out4, en3_out]), name="msfe4_en2_spconv2")
        de4_2_out2 = self.spconv(self.mid_ch, Concatenate(axis=3)([de4_2_out3, en2_out]), name="msfe4_en2_spconv3")
        de4_2_out1 = self.spconv(self.out_ch, Concatenate(axis=3)([de4_2_out2, en1_out]), name="msfe4_en2_spconv4")

        de_out = self.ctfa(de4_2_out1, self.out_ch, name_ta='msfe4_en2_ta', name_fa='msfe4_en2_fa') + en_in

        # Down-sampling
        msfe4_out2 = self.down_sampling(self.out_ch, de_out, name="msfe4_down_sampling2")

        ############################################################################
        # MSFE4(3) - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, msfe4_out2, name="msfe4_en3_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe4_en3_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe4_en3_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe4_en3_conv3")
        en4_out = self.conv(self.mid_ch, en3_out, name="msfe4_en3_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        en_lstm = self.msfe4_en_lstm_3(reshape_out)
        en_lstm = self.msfe4_en_dense_3(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de4_3_out4 = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en4_out]), name="msfe4_en3_spconv1")
        de4_3_out3 = self.spconv(self.mid_ch, Concatenate(axis=3)([de4_3_out4, en3_out]), name="msfe4_en3_spconv2")
        de4_3_out2 = self.spconv(self.mid_ch, Concatenate(axis=3)([de4_3_out3, en2_out]), name="msfe4_en3_spconv3")
        de4_3_out1 = self.spconv(self.out_ch, Concatenate(axis=3)([de4_3_out2, en1_out]), name="msfe4_en3_spconv4")

        de_out = self.ctfa(de4_3_out1, self.out_ch, name_ta='msfe4_en3_ta', name_fa='msfe4_en3_fa') + en_in

        # Down-sampling
        msfe4_out3 = self.down_sampling(self.out_ch, de_out, name="msfe4_down_sampling3")

        ############################################################################
        # MSFE3 - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, msfe4_out3, name="msfe3_en_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe3_en_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe3_en_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe3_en_conv3")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en3_out)
        en_lstm = self.msfe3_en_lstm(reshape_out)
        en_lstm = self.msfe3_en_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de3_out3 = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en3_out]), name="msfe3_en_spconv1")
        de3_out2 = self.spconv(self.mid_ch, Concatenate(axis=3)([de3_out3, en2_out]), name="msfe3_en_spconv2")
        de3_out1 = self.spconv(self.out_ch, Concatenate(axis=3)([de3_out2, en1_out]), name="msfe3_en_spconv3")

        de_out = self.ctfa(de3_out1, self.out_ch, name_ta='msfe3_en_ta', name_fa='msfe3_en_fa') + en_in

        # Down-sampling
        msfe3_out = self.down_sampling(self.out_ch, de_out, name="msfe3_down_sampling")

        ############################################################################
        # LSTM
        ############################################################################
        reshape_out, shape = self.reshape_before_lstm(msfe3_out)
        lstm_out = self.lstm(reshape_out)
        lstm_out = self.dense(lstm_out)
        lstm_out = self.reshape_after_lstm(lstm_out, shape)

        ############################################################################
        # MSFE3 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([lstm_out, msfe3_out]),
                                 name="msfe3_upsampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe3_de_in")

        en1_out = self.conv(self.mid_ch, Concatenate(axis=3)([en_in, de3_out1]), name="msfe3_de_conv1")
        en2_out = self.conv(self.mid_ch, Concatenate(axis=3)([en1_out, de3_out2]), name="msfe3_de_conv2")
        en3_out = self.conv(self.mid_ch, Concatenate(axis=3)([en2_out, de3_out3]), name="msfe3_de_conv3")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en3_out)
        de_lstm = self.msfe3_de_lstm(reshape_out)
        de_lstm = self.msfe3_de_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_lstm, en3_out]), name="msfe3_de_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe3_de_spconv2")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe3_de_spconv3")

        de_out = self.ctfa(de_out, self.out_ch, name_ta='msfe3_de_ta', name_fa='msfe3_de_fa') + en_in

        ############################################################################
        # MSFE4 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out3]),
                                 name="msfe4_upsampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de_in")
        en1_out = self.conv(self.mid_ch, Concatenate(axis=3)([en_in, de4_3_out1]), name="msfe4_de_conv1")
        en2_out = self.conv(self.mid_ch, Concatenate(axis=3)([en1_out, de4_3_out2]), name="msfe4_de_conv2")
        en3_out = self.conv(self.mid_ch, Concatenate(axis=3)([en2_out, de4_3_out3]), name="msfe4_de_conv3")
        en4_out = self.conv(self.mid_ch, Concatenate(axis=3)([en3_out, de4_3_out4]), name="msfe4_de_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        de_lstm = self.msfe4_de_lstm_1(reshape_out)
        de_lstm = self.msfe4_de_dense_1(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_lstm, en4_out]), name="msfe4_de_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe4_de_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe4_de_spconv3")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe4_de_spconv4")

        de_out = self.ctfa(de_out, self.out_ch, name_ta='msfe4_de_ta', name_fa='msfe4_de_fa') + en_in

        ############################################################################
        # MSFE4(2) - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out2]),
                                 name="msfe4_upsampling2")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de2_in")
        en1_out = self.conv(self.mid_ch, Concatenate(axis=3)([en_in, de4_2_out1]), name="msfe4_de2_conv1")
        en2_out = self.conv(self.mid_ch, Concatenate(axis=3)([en1_out, de4_2_out2]), name="msfe4_de2_conv2")
        en3_out = self.conv(self.mid_ch, Concatenate(axis=3)([en2_out, de4_2_out3]), name="msfe4_de2_conv3")
        en4_out = self.conv(self.mid_ch, Concatenate(axis=3)([en3_out, de4_2_out4]), name="msfe4_de2_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        de_lstm = self.msfe4_de_lstm_2(reshape_out)
        de_lstm = self.msfe4_de_dense_2(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_lstm, en4_out]), name="msfe4_de2_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe4_de2_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe4_de2_spconv3")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe4_de2_spconv4")

        de_out = self.ctfa(de_out, self.out_ch, name_ta='msfe4_de2_ta', name_fa='msfe4_de2_fa') + en_in

        ############################################################################
        # MSFE4(3) - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out]),
                                 name="msfe4_upsampling3")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de3_in")
        en1_out = self.conv(self.mid_ch, Concatenate(axis=3)([en_in, de4_out1]), name="msfe4_de3_conv1")
        en2_out = self.conv(self.mid_ch, Concatenate(axis=3)([en1_out, de4_out2]), name="msfe4_de3_conv2")
        en3_out = self.conv(self.mid_ch, Concatenate(axis=3)([en2_out, de4_out3]), name="msfe4_de3_conv3")
        en4_out = self.conv(self.mid_ch, Concatenate(axis=3)([en3_out, de4_out4]), name="msfe4_de3_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        de_lstm = self.msfe4_de_lstm_3(reshape_out)
        de_lstm = self.msfe4_de_dense_3(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_lstm, en4_out]), name="msfe4_de3_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe4_de3_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe4_de3_spconv3")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe4_de3_spconv4")

        de_out = self.ctfa(de_out, self.out_ch, name_ta='msfe4_de3_ta', name_fa='msfe4_de3_fa') + en_in

        ############################################################################
        # MSFE5 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe5_out]), name="msfe5_upsampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe5_de_in")
        en1_out = self.conv(self.mid_ch, Concatenate(axis=3)([en_in, de5_out1]), name="msfe5_de_conv1")
        en2_out = self.conv(self.mid_ch, Concatenate(axis=3)([en1_out, de5_out2]), name="msfe5_de_conv2")
        en3_out = self.conv(self.mid_ch, Concatenate(axis=3)([en2_out, de5_out3]), name="msfe5_de_conv3")
        en4_out = self.conv(self.mid_ch, Concatenate(axis=3)([en3_out, de5_out4]), name="msfe5_de_conv4")
        en5_out = self.conv(self.mid_ch, Concatenate(axis=3)([en4_out, de5_out5]), name="msfe5_de_conv5")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en5_out)
        de_lstm = self.msfe5_de_lstm(reshape_out)
        de_lstm = self.msfe5_de_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_lstm, en5_out]), name="msfe5_de_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en4_out]), name="msfe5_de_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe5_de_spconv3")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe5_de_spconv4")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe5_de_spconv5")

        de_out = self.ctfa(de_out, self.out_ch, name_ta='msfe5_de_ta', name_fa='msfe5_de_fa') + en_in

        ############################################################################
        # MSFE6 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe6_out]), name="msfe6_upsampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe6_de_in")
        en1_out = self.conv(self.mid_ch, Concatenate(axis=3)([en_in, de6_out1]), name="msfe6_de_conv1")
        en2_out = self.conv(self.mid_ch, Concatenate(axis=3)([en1_out, de6_out2]), name="msfe6_de_conv2")
        en3_out = self.conv(self.mid_ch, Concatenate(axis=3)([en2_out, de6_out3]), name="msfe6_de_conv3")
        en4_out = self.conv(self.mid_ch, Concatenate(axis=3)([en3_out, de6_out4]), name="msfe6_de_conv4")
        en5_out = self.conv(self.mid_ch, Concatenate(axis=3)([en4_out, de6_out5]), name="msfe6_de_conv5")
        en6_out = self.conv(self.mid_ch, Concatenate(axis=3)([en5_out, de6_out6]), name="msfe6_de_conv6")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en6_out)
        de_lstm = self.msfe6_de_lstm(reshape_out)
        de_lstm = self.msfe6_de_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_lstm, en6_out]), name="msfe6_de_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en5_out]), name="msfe6_de_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en4_out]), name="msfe6_de_spconv3")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe6_de_spconv4")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe6_de_spconv5")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe6_de_spconv6")

        de_out = self.ctfa(de_out, self.out_ch, name_ta='msfe6_de_ta', name_fa='msfe6_de_fa') + en_in

        out = self.out_conv(de_out)

        out = tf.squeeze(out, axis=3)
        paddings = tf.constant([[0, 0], [0, 0], [1, 0]])
        est_mags = tf.pad(out, paddings, mode='CONSTANT')  # Direct Mapping

        s1_stft = (tf.cast(est_mags, tf.complex64) * tf.exp((1j * tf.cast(phase, tf.complex64))))
        y = tf.signal.inverse_stft(s1_stft, self.win_len, self.hop_len, fft_length=self.fft_len,
                                   window_fn=tf.signal.inverse_stft_window_fn(self.hop_len))

        return y

    def build_model(self):
        if self.opt.chunk_size % self.opt.hop_len == 0:
            x = Input(batch_shape=(None, self.opt.chunk_size))  # 8ms
        else:
            x = Input(batch_shape=(None, self.opt.chunk_size + (self.opt.hop_len // 2)))  # 16ms

        y = self.train_model(x)

        self.model = Model(inputs=x, outputs=y)

        return self.model

    def tflite_model(self):
        x = Input(batch_shape=(1, 1, 256, 1))

        en_in = self.inconv(self.out_ch, x, name="input_layer")
        ############################################################################
        # MSFE6 - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, en_in, name="msfe6_en_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en_in)
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe6_en_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en1_out)
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe6_en_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en2_out)
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe6_en_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en3_out)
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe6_en_conv4")
        en4_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en4_out)
        en5_out = self.conv_valid(self.mid_ch, en4_out_, name="msfe6_en_conv5")
        en5_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en5_out)
        en6_out = self.conv_valid(self.mid_ch, en5_out_, name="msfe6_en_conv6")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en6_out)
        en_lstm, _, _ = self.msfe6_en_lstm_tfl(reshape_out)
        en_lstm = self.msfe6_en_dense_tfl(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de6_out6 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([en_lstm, en6_out])),
                                     name="msfe6_en_spconv1")
        de6_out5 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([de6_out6, en5_out])),
                                     name="msfe6_en_spconv2")
        de6_out4 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([de6_out5, en4_out])),
                                     name="msfe6_en_spconv3")
        de6_out3 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([de6_out4, en3_out])),
                                     name="msfe6_en_spconv4")
        de6_out2 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([de6_out3, en2_out])),
                                     name="msfe6_en_spconv5")
        de6_out1 = self.spconv_valid(self.out_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([de6_out2, en1_out])),
                                     name="msfe6_en_spconv6")

        de_out = self.ctfa_rt(de6_out1, self.out_ch, name_ta='msfe6_en_ta', name_fa='msfe6_en_fa') + en_in

        # Down-sampling
        msfe6_out = self.down_sampling(self.out_ch, de_out, name="msfe6_down_sampling")  # [None, 372, 128 ,64]

        ############################################################################
        # MSFE5 - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, msfe6_out, name="msfe5_en_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en_in)
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe5_en_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en1_out)
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe5_en_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en2_out)
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe5_en_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en3_out)
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe5_en_conv4")
        en4_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en4_out)
        en5_out = self.conv_valid(self.mid_ch, en4_out_, name="msfe5_en_conv5")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en5_out)
        en_lstm, _, _ = self.msfe5_en_lstm_tfl(reshape_out)
        en_lstm = self.msfe5_en_dense_tfl(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de5_out5 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([en_lstm, en5_out])),
                                     name="msfe5_en_spconv1")
        de5_out4 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([de5_out5, en4_out])),
                                     name="msfe5_en_spconv2")
        de5_out3 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([de5_out4, en3_out])),
                                     name="msfe5_en_spconv3")
        de5_out2 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([de5_out3, en2_out])),
                                     name="msfe5_en_spconv4")
        de5_out1 = self.spconv_valid(self.out_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([de5_out2, en1_out])),
                                     name="msfe5_en_spconv5")

        de_out = self.ctfa_rt(de5_out1, self.out_ch, name_ta='msfe5_en_ta', name_fa='msfe5_en_fa') + en_in

        # Down-sampling
        msfe5_out = self.down_sampling(self.out_ch, de_out, name="msfe5_down_sampling")  # [None, 372, 128 ,64]

        ############################################################################
        # MSFE4 - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, msfe5_out, name="msfe4_en_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en_in)
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe4_en_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en1_out)
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe4_en_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en2_out)
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe4_en_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en3_out)
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe4_en_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        en_lstm, _, _ = self.msfe4_en_lstm_1_tfl(reshape_out)
        en_lstm = self.msfe4_en_dense_1_tfl(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de4_out4 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([en_lstm, en4_out])),
                                     name="msfe4_en_spconv1")
        de4_out3 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([de4_out4, en3_out])),
                                     name="msfe4_en_spconv2")
        de4_out2 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([de4_out3, en2_out])),
                                     name="msfe4_en_spconv3")
        de4_out1 = self.spconv_valid(self.out_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([de4_out2, en1_out])),
                                     name="msfe4_en_spconv4")

        de_out = self.ctfa_rt(de4_out1, self.out_ch, name_ta='msfe4_en_ta', name_fa='msfe4_en_fa') + en_in

        # Down-sampling
        msfe4_out = self.down_sampling(self.out_ch, de_out, name="msfe4_down_sampling")  # [None, 372, 128 ,64]

        ############################################################################
        # MSFE4(2) - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, msfe4_out, name="msfe4_en2_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en_in)
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe4_en2_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en1_out)
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe4_en2_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en2_out)
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe4_en2_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en3_out)
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe4_en2_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        en_lstm, _, _ = self.msfe4_en_lstm_2_tfl(reshape_out)
        en_lstm = self.msfe4_en_dense_2_tfl(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de4_2_out4 = self.spconv_valid(self.mid_ch,
                                       ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                           Concatenate(axis=3)([en_lstm, en4_out])),
                                       name="msfe4_en2_spconv1")
        de4_2_out3 = self.spconv_valid(self.mid_ch,
                                       ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                           Concatenate(axis=3)([de4_2_out4, en3_out])),
                                       name="msfe4_en2_spconv2")
        de4_2_out2 = self.spconv_valid(self.mid_ch,
                                       ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                           Concatenate(axis=3)([de4_2_out3, en2_out])),
                                       name="msfe4_en2_spconv3")
        de4_2_out1 = self.spconv_valid(self.out_ch,
                                       ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                           Concatenate(axis=3)([de4_2_out2, en1_out])),
                                       name="msfe4_en2_spconv4")

        de_out = self.ctfa_rt(de4_2_out1, self.out_ch, name_ta='msfe4_en2_ta', name_fa='msfe4_en2_fa') + en_in

        # Down-sampling
        msfe4_out2 = self.down_sampling(self.out_ch, de_out, name="msfe4_down_sampling2")  # [None, 372, 128 ,64]

        ############################################################################
        # MSFE4(3) - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, msfe4_out2, name="msfe4_en3_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en_in)
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe4_en3_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en1_out)
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe4_en3_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en2_out)
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe4_en3_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en3_out)
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe4_en3_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        en_lstm, _, _ = self.msfe4_en_lstm_3_tfl(reshape_out)
        en_lstm = self.msfe4_en_dense_3_tfl(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de4_3_out4 = self.spconv_valid(self.mid_ch,
                                       ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                           Concatenate(axis=3)([en_lstm, en4_out])),
                                       name="msfe4_en3_spconv1")
        de4_3_out3 = self.spconv_valid(self.mid_ch,
                                       ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                           Concatenate(axis=3)([de4_3_out4, en3_out])),
                                       name="msfe4_en3_spconv2")
        de4_3_out2 = self.spconv_valid(self.mid_ch,
                                       ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                           Concatenate(axis=3)([de4_3_out3, en2_out])),
                                       name="msfe4_en3_spconv3")
        de4_3_out1 = self.spconv_valid(self.out_ch,
                                       ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                           Concatenate(axis=3)([de4_3_out2, en1_out])),
                                       name="msfe4_en3_spconv4")

        de_out = self.ctfa_rt(de4_3_out1, self.out_ch, name_ta='msfe4_en3_ta', name_fa='msfe4_en3_fa') + en_in

        # Down-sampling
        msfe4_out3 = self.down_sampling(self.out_ch, de_out, name="msfe4_down_sampling3")  # [None, 372, 128 ,64]

        ############################################################################
        # MSFE3 - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, msfe4_out3, name="msfe3_en_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en_in)
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe3_en_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en1_out)
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe3_en_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en2_out)
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe3_en_conv3")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en3_out)
        en_lstm, _, _ = self.msfe3_en_lstm_tfl(reshape_out)
        en_lstm = self.msfe3_en_dense_tfl(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de3_out3 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([en_lstm, en3_out])),
                                     name="msfe3_en_spconv1")
        de3_out2 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([de3_out3, en2_out])),
                                     name="msfe3_en_spconv2")
        de3_out1 = self.spconv_valid(self.out_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(
                                         Concatenate(axis=3)([de3_out2, en1_out])),
                                     name="msfe3_en_spconv3")

        de_out = self.ctfa_rt(de3_out1, self.out_ch, name_ta='msfe3_en_ta', name_fa='msfe3_en_fa') + en_in

        # Down-sampling
        msfe3_out = self.down_sampling(self.out_ch, de_out, name="msfe3_down_sampling")  # [None, 372, 128 ,64]

        ############################################################################
        # LSTM
        ############################################################################
        reshape_out, shape = self.reshape_before_lstm(msfe3_out)
        lstm_out, _, _ = self.lstm_tfl(reshape_out)
        lstm_out = self.dense_tfl(lstm_out)
        lstm_out = self.reshape_after_lstm(lstm_out, shape)

        ############################################################################
        # MSFE3 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([lstm_out, msfe3_out]),
                                 name="msfe3_upsampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe3_de_in")

        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en_in, de3_out1]))
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe3_de_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en1_out, de3_out2]))
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe3_de_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en2_out, de3_out3]))
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe3_de_conv3")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en3_out)
        de_lstm, _, _ = self.msfe3_de_lstm_tfl(reshape_out)
        de_lstm = self.msfe3_de_dense_tfl(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_lstm, en3_out])),
                                   name="msfe3_de_spconv1")
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en2_out])),
                                   name="msfe3_de_spconv2")
        de_out = self.spconv_valid(self.out_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en1_out])),
                                   name="msfe3_de_spconv3")

        de_out = self.ctfa_rt(de_out, self.out_ch, name_ta='msfe3_de_ta', name_fa='msfe3_de_fa') + en_in

        ############################################################################
        # MSFE4 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out3]),
                                 name="msfe4_upsampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en_in, de4_3_out1]))
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe4_de_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en1_out, de4_3_out2]))
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe4_de_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en2_out, de4_3_out3]))
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe4_de_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en3_out, de4_3_out4]))
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe4_de_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        de_lstm, _, _ = self.msfe4_de_lstm_1_tfl(reshape_out)
        de_lstm = self.msfe4_de_dense_1_tfl(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_lstm, en4_out])),
                                   name="msfe4_de_spconv1")
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en3_out])),
                                   name="msfe4_de_spconv2")
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en2_out])),
                                   name="msfe4_de_spconv3")
        de_out = self.spconv_valid(self.out_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en1_out])),
                                   name="msfe4_de_spconv4")

        de_out = self.ctfa_rt(de_out, self.out_ch, name_ta='msfe4_de_ta', name_fa='msfe4_de_fa') + en_in

        ############################################################################
        # MSFE4(2) - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out2]),
                                 name="msfe4_upsampling2")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de2_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en_in, de4_2_out1]))
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe4_de2_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en1_out, de4_2_out2]))
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe4_de2_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en2_out, de4_2_out3]))
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe4_de2_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en3_out, de4_2_out4]))
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe4_de2_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        de_lstm, _, _ = self.msfe4_de_lstm_2_tfl(reshape_out)
        de_lstm = self.msfe4_de_dense_2_tfl(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_lstm, en4_out])),
                                   name="msfe4_de2_spconv1")
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en3_out])),
                                   name="msfe4_de2_spconv2")
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en2_out])),
                                   name="msfe4_de2_spconv3")
        de_out = self.spconv_valid(self.out_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en1_out])),
                                   name="msfe4_de2_spconv4")

        de_out = self.ctfa_rt(de_out, self.out_ch, name_ta='msfe4_de2_ta', name_fa='msfe4_de2_fa') + en_in

        ############################################################################
        # MSFE4(3) - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out]),
                                 name="msfe4_upsampling3")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de3_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en_in, de4_out1]))
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe4_de3_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en1_out, de4_out2]))
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe4_de3_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en2_out, de4_out3]))
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe4_de3_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en3_out, de4_out4]))
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe4_de3_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        de_lstm, _, _ = self.msfe4_de_lstm_3_tfl(reshape_out)
        de_lstm = self.msfe4_de_dense_3_tfl(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_lstm, en4_out])),
                                   name="msfe4_de3_spconv1")
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en3_out])),
                                   name="msfe4_de3_spconv2")
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en2_out])),
                                   name="msfe4_de3_spconv3")
        de_out = self.spconv_valid(self.out_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en1_out])),
                                   name="msfe4_de3_spconv4")

        de_out = self.ctfa_rt(de_out, self.out_ch, name_ta='msfe4_de3_ta', name_fa='msfe4_de3_fa') + en_in

        ############################################################################
        # MSFE5 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe5_out]), name="msfe5_upsampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe5_de_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en_in, de5_out1]))
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe5_de_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en1_out, de5_out2]))
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe5_de_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en2_out, de5_out3]))
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe5_de_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en3_out, de5_out4]))
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe5_de_conv4")
        en4_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en4_out, de5_out5]))
        en5_out = self.conv_valid(self.mid_ch, en4_out_, name="msfe5_de_conv5")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en5_out)
        de_lstm, _, _ = self.msfe5_de_lstm_tfl(reshape_out)
        de_lstm = self.msfe5_de_dense_tfl(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_lstm, en5_out])),
                                   name="msfe5_de_spconv1")
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en4_out])),
                                   name="msfe5_de_spconv2")
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en3_out])),
                                   name="msfe5_de_spconv3")
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en2_out])),
                                   name="msfe5_de_spconv4")
        de_out = self.spconv_valid(self.out_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en1_out])),
                                   name="msfe5_de_spconv5")

        de_out = self.ctfa_rt(de_out, self.out_ch, name_ta='msfe5_de_ta', name_fa='msfe5_de_fa') + en_in

        ############################################################################
        # MSFE6 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe6_out]), name="msfe6_upsampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe6_de_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en_in, de6_out1]))
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe6_de_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en1_out, de6_out2]))
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe6_de_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en2_out, de6_out3]))
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe6_de_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en3_out, de6_out4]))
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe6_de_conv4")
        en4_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en4_out, de6_out5]))
        en5_out = self.conv_valid(self.mid_ch, en4_out_, name="msfe6_de_conv5")
        en5_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en5_out, de6_out6]))
        en6_out = self.conv_valid(self.mid_ch, en5_out_, name="msfe6_de_conv6")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en6_out)
        de_lstm, _, _ = self.msfe6_de_lstm_tfl(reshape_out)
        de_lstm = self.msfe6_de_dense_tfl(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_lstm, en6_out])),
                                   name="msfe6_de_spconv1")
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en5_out])),
                                   name="msfe6_de_spconv2")
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en4_out])),
                                   name="msfe6_de_spconv3")
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en3_out])),
                                   name="msfe6_de_spconv4")
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en2_out])),
                                   name="msfe6_de_spconv5")
        de_out = self.spconv_valid(self.out_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de_out, en1_out])),
                                   name="msfe6_de_spconv6")

        de_out = self.ctfa_rt(de_out, self.out_ch, name_ta='msfe6_de_ta', name_fa='msfe6_de_fa') + en_in

        y = self.out_conv(de_out)

        model = Model(inputs=x, outputs=y)
        model.summary()

        return model
