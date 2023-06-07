'''
for real-time speech enhancement
baseline (Nested U-Net)
'''

import keras
from keras import Model, Input
import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, LayerNormalization, PReLU, Lambda, Concatenate, Reshape, Permute, \
    ZeroPadding2D, ReLU, Conv1D, Activation, GlobalAveragePooling1D, AveragePooling1D
import numpy as np

class NUTLS(keras.Model):
    def __init__(self, opt):
        super(NUTLS, self).__init__()
        self.in_ch = 1
        self.mid_ch = 32
        self.out_ch = 64
        self.win_len = opt.win_len
        self.fft_len = opt.fft_len
        self.hop_len = opt.hop_len
        self.opt = opt
        self.out_conv = Conv2D(self.in_ch, kernel_size=1)

    ############################################################################
    # Tools
    ############################################################################
    def inconv(self, out_ch, x, name=None):
        in_conv = keras.Sequential([
            Conv2D(out_ch, kernel_size=1),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return in_conv(x)

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

    # CTFA (Causal Time Frequency Attention)
    def ctfa(self, x, in_ch, out_ch=16, time_seq=32, name_ta=None, name_fa=None):
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
            Lambda(lambda x: tf.broadcast_to(x, shape=[self.opt.batch_size, T, F, C]))

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

    ############################################################################
    # for dilated dense block
    ############################################################################
    def dilated_dense_block_in(self, in_ch, x, name=None):
        dd_block_in = keras.Sequential([
            ZeroPadding2D(padding=((1, 0), (1, 1))),
            Conv2D(in_ch // 2, kernel_size=(2, 3), strides=1, padding='valid'),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_in(x)

    def dilated_dense_block_1(self, in_ch, x, name=None):
        dd_block_1 = keras.Sequential([
            ZeroPadding2D(padding=((1, 0), (1, 1))),
            Conv2D(in_ch // 2, kernel_size=(2, 3), padding='valid', dilation_rate=1, groups=in_ch // 2),
            Conv2D(in_ch // 2, kernel_size=1),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_1(x)

    def dilated_dense_block_2(self, in_ch, x, name=None):
        dd_block_2 = keras.Sequential([
            ZeroPadding2D(padding=((2, 0), (2, 2))),
            Conv2D(in_ch // 2, kernel_size=(2, 3), padding='valid', dilation_rate=2, groups=in_ch // 2),
            Conv2D(in_ch // 2, kernel_size=1),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_2(x)

    def dilated_dense_block_3(self, in_ch, x, name=None):
        dd_block_3 = keras.Sequential([
            ZeroPadding2D(padding=((4, 0), (4, 4))),
            Conv2D(in_ch // 2, kernel_size=(2, 3), padding='valid', dilation_rate=4, groups=in_ch // 2),
            Conv2D(in_ch // 2, kernel_size=1),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_3(x)

    def dilated_dense_block_4(self, in_ch, x, name=None):
        dd_block_4 = keras.Sequential([
            ZeroPadding2D(padding=((8, 0), (8, 8))),
            Conv2D(in_ch // 2, kernel_size=(2, 3), padding='valid', dilation_rate=8, groups=in_ch // 2),
            Conv2D(in_ch // 2, kernel_size=1),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_4(x)

    def dilated_dense_block_5(self, in_ch, x, name=None):
        dd_block_5 = keras.Sequential([
            ZeroPadding2D(padding=((16, 0), (16, 16))),
            Conv2D(in_ch // 2, kernel_size=(2, 3), padding='valid', dilation_rate=16, groups=in_ch // 2),
            Conv2D(in_ch // 2, kernel_size=1),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_5(x)

    def dilated_dense_block_6(self, in_ch, x, name=None):
        dd_block_6 = keras.Sequential([
            ZeroPadding2D(padding=((32, 0), (32, 32))),
            Conv2D(in_ch // 2, kernel_size=(2, 3), padding='valid', dilation_rate=32, groups=in_ch // 2),
            Conv2D(in_ch // 2, kernel_size=1),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_6(x)

    def dilated_dense_block_out(self, out_ch, x, name=None):
        dd_block_out = keras.Sequential([
            ZeroPadding2D(padding=((1, 0), (1, 1))),
            Conv2D(out_ch, kernel_size=(2, 3), strides=1, padding='valid'),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_out(x)

    ############################################################################
    # Dilated dense block for TFLite
    ############################################################################
    def dilated_dense_block_in_valid(self, in_ch, x, name=None):
        dd_block_in = keras.Sequential([
            ZeroPadding2D(padding=((0, 0), (1, 1))),
            Conv2D(in_ch // 2, kernel_size=(2, 3), strides=1, padding='valid'),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_in(x)

    def dilated_dense_block_1_valid(self, in_ch, x, name=None):
        dd_block_1 = keras.Sequential([
            ZeroPadding2D(padding=((0, 0), (1, 1))),
            Conv2D(in_ch // 2, kernel_size=(2, 3), padding='valid', dilation_rate=1, groups=in_ch // 2),
            Conv2D(in_ch // 2, kernel_size=1),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_1(x)

    def dilated_dense_block_2_valid(self, in_ch, x, name=None):
        dd_block_2 = keras.Sequential([
            ZeroPadding2D(padding=((0, 0), (2, 2))),
            Conv2D(in_ch // 2, kernel_size=(2, 3), padding='valid', dilation_rate=2, groups=in_ch // 2),
            Conv2D(in_ch // 2, kernel_size=1),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_2(x)

    def dilated_dense_block_3_valid(self, in_ch, x, name=None):
        dd_block_3 = keras.Sequential([
            ZeroPadding2D(padding=((0, 0), (4, 4))),
            Conv2D(in_ch // 2, kernel_size=(2, 3), padding='valid', dilation_rate=4, groups=in_ch // 2),
            Conv2D(in_ch // 2, kernel_size=1),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_3(x)

    def dilated_dense_block_4_valid(self, in_ch, x, name=None):
        dd_block_4 = keras.Sequential([
            ZeroPadding2D(padding=((0, 0), (8, 8))),
            Conv2D(in_ch // 2, kernel_size=(2, 3), padding='valid', dilation_rate=8, groups=in_ch // 2),
            Conv2D(in_ch // 2, kernel_size=1),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_4(x)

    def dilated_dense_block_5_valid(self, in_ch, x, name=None):
        dd_block_5 = keras.Sequential([
            ZeroPadding2D(padding=((0, 0), (16, 16))),
            Conv2D(in_ch // 2, kernel_size=(2, 3), padding='valid', dilation_rate=16, groups=in_ch // 2),
            Conv2D(in_ch // 2, kernel_size=1),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_5(x)

    def dilated_dense_block_6_valid(self, in_ch, x, name=None):
        dd_block_6 = keras.Sequential([
            ZeroPadding2D(padding=((0, 0), (32, 32))),
            Conv2D(in_ch // 2, kernel_size=(2, 3), padding='valid', dilation_rate=32, groups=in_ch // 2),
            Conv2D(in_ch // 2, kernel_size=1),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_6(x)

    def dilated_dense_block_out_valid(self, out_ch, x, name=None):
        dd_block_out = keras.Sequential([
            ZeroPadding2D(padding=((0, 0), (1, 1))),
            Conv2D(out_ch, kernel_size=(2, 3), strides=1, padding='valid'),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return dd_block_out(x)

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

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in(self.mid_ch, en6_out, name="msfe6_en_ddb_in")
        ddb_out1 = self.dilated_dense_block_1(self.mid_ch, ddb_out0, name="msfe6_en_ddb1")
        ddb_out2 = self.dilated_dense_block_2(self.mid_ch, Concatenate(axis=3)([ddb_out1, ddb_out0]),
                                              name="msfe6_en_ddb2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3(self.mid_ch, Concatenate(axis=3)([ddb_out3, ddb_out0]),
                                              name="msfe6_en_ddb3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4(self.mid_ch, Concatenate(axis=3)([ddb_out4, ddb_out0]),
                                              name="msfe6_en_ddb4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5(self.mid_ch, Concatenate(axis=3)([ddb_out5, ddb_out0]),
                                              name="msfe6_en_ddb5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6(self.mid_ch, Concatenate(axis=3)([ddb_out6, ddb_out0]),
                                              name="msfe6_en_ddb6")

        ddb_out = self.dilated_dense_block_out(self.mid_ch, ddb_out6, name="msfe6_en_ddb_out")

        # Decoder
        de6_out6 = self.spconv(self.mid_ch, Concatenate(axis=3)([ddb_out, en6_out]), name="msfe6_en_spconv1")
        de6_out5 = self.spconv(self.mid_ch, Concatenate(axis=3)([de6_out6, en5_out]), name="msfe6_en_spconv2")
        de6_out4 = self.spconv(self.mid_ch, Concatenate(axis=3)([de6_out5, en4_out]), name="msfe6_en_spconv3")
        de6_out3 = self.spconv(self.mid_ch, Concatenate(axis=3)([de6_out4, en3_out]), name="msfe6_en_spconv4")
        de6_out2 = self.spconv(self.mid_ch, Concatenate(axis=3)([de6_out3, en2_out]), name="msfe6_en_spconv5")
        de6_out1 = self.spconv(self.out_ch, Concatenate(axis=3)([de6_out2, en1_out]), name="msfe6_en_spconv6")

        de_out = self.ctfa(de6_out1, self.out_ch, name_ta='msfe6_en_ta', name_fa='msfe6_en_fa') + en_in

        # Down-sampling
        msfe6_out = self.down_sampling(self.out_ch, de_out, name="msfe6_down_sampling")  # [None, 372, 128 ,64]

        ############################################################################
        # MSFE5 - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, msfe6_out, name="msfe5_en_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe5_en_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe5_en_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe5_en_conv3")
        en4_out = self.conv(self.mid_ch, en3_out, name="msfe5_en_conv4")
        en5_out = self.conv(self.mid_ch, en4_out, name="msfe5_en_conv5")

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in(self.mid_ch, en5_out, name="msfe5_en_ddb_in")
        ddb_out1 = self.dilated_dense_block_1(self.mid_ch, ddb_out0, name="msfe5_en_ddb1")
        ddb_out2 = self.dilated_dense_block_2(self.mid_ch, Concatenate(axis=3)([ddb_out1, ddb_out0]),
                                              name="msfe5_en_ddb2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3(self.mid_ch, Concatenate(axis=3)([ddb_out3, ddb_out0]),
                                              name="msfe5_en_ddb3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4(self.mid_ch, Concatenate(axis=3)([ddb_out4, ddb_out0]),
                                              name="msfe5_en_ddb4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5(self.mid_ch, Concatenate(axis=3)([ddb_out5, ddb_out0]),
                                              name="msfe5_en_ddb5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6(self.mid_ch, Concatenate(axis=3)([ddb_out6, ddb_out0]),
                                              name="msfe5_en_ddb6")

        ddb_out = self.dilated_dense_block_out(self.mid_ch, ddb_out6, name="msfe5_en_ddb_out")

        # Decoder
        de5_out5 = self.spconv(self.mid_ch, Concatenate(axis=3)([ddb_out, en5_out]), name="msfe5_en_spconv1")
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

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in(self.mid_ch, en4_out, name="msfe4_en_ddb_in")
        ddb_out1 = self.dilated_dense_block_1(self.mid_ch, ddb_out0, name="msfe4_en_ddb1")
        ddb_out2 = self.dilated_dense_block_2(self.mid_ch, Concatenate(axis=3)([ddb_out1, ddb_out0]),
                                              name="msfe4_en_ddb2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3(self.mid_ch, Concatenate(axis=3)([ddb_out3, ddb_out0]),
                                              name="msfe4_en_ddb3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4(self.mid_ch, Concatenate(axis=3)([ddb_out4, ddb_out0]),
                                              name="msfe4_en_ddb4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5(self.mid_ch, Concatenate(axis=3)([ddb_out5, ddb_out0]),
                                              name="msfe4_en_ddb5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6(self.mid_ch, Concatenate(axis=3)([ddb_out6, ddb_out0]),
                                              name="msfe4_en_ddb6")

        ddb_out = self.dilated_dense_block_out(self.mid_ch, ddb_out6, name="msfe4_en_ddb_out")

        # Decoder
        de4_out4 = self.spconv(self.mid_ch, Concatenate(axis=3)([ddb_out, en4_out]), name="msfe4_en_spconv1")
        de4_out3 = self.spconv(self.mid_ch, Concatenate(axis=3)([de4_out4, en3_out]), name="msfe4_en_spconv2")
        de4_out2 = self.spconv(self.mid_ch, Concatenate(axis=3)([de4_out3, en2_out]), name="msfe4_en_spconv3")
        de4_out1 = self.spconv(self.out_ch, Concatenate(axis=3)([de4_out2, en1_out]), name="msfe4_en_spconv4")

        de_out = self.ctfa(de4_out1, self.out_ch, name_ta='msfe4_en_ta', name_fa='msfe4_en_fa') + en_in

        # Down-sampling
        msfe4_out = self.down_sampling(self.out_ch, de_out, name="msfe4_down_sampling")  # [None, 372, 128 ,64]

        ############################################################################
        # MSFE4(2) - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, msfe4_out, name="msfe4_en2_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe4_en2_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe4_en2_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe4_en2_conv3")
        en4_out = self.conv(self.mid_ch, en3_out, name="msfe4_en2_conv4")

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in(self.mid_ch, en4_out, name="msfe4_en2_ddb_in")
        ddb_out1 = self.dilated_dense_block_1(self.mid_ch, ddb_out0, name="msfe4_en2_ddb1")
        ddb_out2 = self.dilated_dense_block_2(self.mid_ch, Concatenate(axis=3)([ddb_out1, ddb_out0]),
                                              name="msfe4_en2_ddb2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3(self.mid_ch, Concatenate(axis=3)([ddb_out3, ddb_out0]),
                                              name="msfe4_en2_ddb3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4(self.mid_ch, Concatenate(axis=3)([ddb_out4, ddb_out0]),
                                              name="msfe4_en2_ddb4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5(self.mid_ch, Concatenate(axis=3)([ddb_out5, ddb_out0]),
                                              name="msfe4_en2_ddb5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6(self.mid_ch, Concatenate(axis=3)([ddb_out6, ddb_out0]),
                                              name="msfe4_en2_ddb6")

        ddb_out = self.dilated_dense_block_out(self.mid_ch, ddb_out6, name="msfe4_en2_ddb_out")

        # Decoder
        de4_2_out4 = self.spconv(self.mid_ch, Concatenate(axis=3)([ddb_out, en4_out]), name="msfe4_en2_spconv1")
        de4_2_out3 = self.spconv(self.mid_ch, Concatenate(axis=3)([de4_2_out4, en3_out]), name="msfe4_en2_spconv2")
        de4_2_out2 = self.spconv(self.mid_ch, Concatenate(axis=3)([de4_2_out3, en2_out]), name="msfe4_en2_spconv3")
        de4_2_out1 = self.spconv(self.out_ch, Concatenate(axis=3)([de4_2_out2, en1_out]), name="msfe4_en2_spconv4")

        de_out = self.ctfa(de4_2_out1, self.out_ch, name_ta='msfe4_en2_ta', name_fa='msfe4_en2_fa') + en_in

        # Down-sampling
        msfe4_out2 = self.down_sampling(self.out_ch, de_out, name="msfe4_down_sampling2")  # [None, 372, 128 ,64]

        ############################################################################
        # MSFE4(3) - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, msfe4_out2, name="msfe4_en3_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe4_en3_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe4_en3_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe4_en3_conv3")
        en4_out = self.conv(self.mid_ch, en3_out, name="msfe4_en3_conv4")

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in(self.mid_ch, en4_out, name="msfe4_en3_ddb_in")
        ddb_out1 = self.dilated_dense_block_1(self.mid_ch, ddb_out0, name="msfe4_en3_ddb1")
        ddb_out2 = self.dilated_dense_block_2(self.mid_ch, Concatenate(axis=3)([ddb_out1, ddb_out0]),
                                              name="msfe4_en3_ddb2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3(self.mid_ch, Concatenate(axis=3)([ddb_out3, ddb_out0]),
                                              name="msfe4_en3_ddb3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4(self.mid_ch, Concatenate(axis=3)([ddb_out4, ddb_out0]),
                                              name="msfe4_en3_ddb4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5(self.mid_ch, Concatenate(axis=3)([ddb_out5, ddb_out0]),
                                              name="msfe4_en3_ddb5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6(self.mid_ch, Concatenate(axis=3)([ddb_out6, ddb_out0]),
                                              name="msfe4_en3_ddb6")

        ddb_out = self.dilated_dense_block_out(self.mid_ch, ddb_out6, name="msfe4_en3_ddb_out")

        # Decoder
        de4_3_out4 = self.spconv(self.mid_ch, Concatenate(axis=3)([ddb_out, en4_out]), name="msfe4_en3_spconv1")
        de4_3_out3 = self.spconv(self.mid_ch, Concatenate(axis=3)([de4_3_out4, en3_out]), name="msfe4_en3_spconv2")
        de4_3_out2 = self.spconv(self.mid_ch, Concatenate(axis=3)([de4_3_out3, en2_out]), name="msfe4_en3_spconv3")
        de4_3_out1 = self.spconv(self.out_ch, Concatenate(axis=3)([de4_3_out2, en1_out]), name="msfe4_en3_spconv4")

        de_out = self.ctfa(de4_3_out1, self.out_ch, name_ta='msfe4_en3_ta', name_fa='msfe4_en3_fa') + en_in

        # Down-sampling
        msfe4_out3 = self.down_sampling(self.out_ch, de_out, name="msfe4_down_sampling3")  # [None, 372, 128 ,64]

        ############################################################################
        # MSFE3 - Encoder
        ############################################################################
        en_in = self.inconv(self.out_ch, msfe4_out3, name="msfe3_en_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe3_en_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe3_en_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe3_en_conv3")

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in(self.mid_ch, en3_out, name="msfe3_en_ddb_in")
        ddb_out1 = self.dilated_dense_block_1(self.mid_ch, ddb_out0, name="msfe3_en_ddb1")
        ddb_out2 = self.dilated_dense_block_2(self.mid_ch, Concatenate(axis=3)([ddb_out1, ddb_out0]),
                                              name="msfe3_en_ddb2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3(self.mid_ch, Concatenate(axis=3)([ddb_out3, ddb_out0]),
                                              name="msfe3_en_ddb3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4(self.mid_ch, Concatenate(axis=3)([ddb_out4, ddb_out0]),
                                              name="msfe3_en_ddb4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5(self.mid_ch, Concatenate(axis=3)([ddb_out5, ddb_out0]),
                                              name="msfe3_en_ddb5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6(self.mid_ch, Concatenate(axis=3)([ddb_out6, ddb_out0]),
                                              name="msfe3_en_ddb6")

        ddb_out = self.dilated_dense_block_out(self.mid_ch, ddb_out6, name="msfe3_en_ddb_out")

        # Decoder
        de3_out3 = self.spconv(self.mid_ch, Concatenate(axis=3)([ddb_out, en3_out]), name="msfe3_en_spconv1")
        de3_out2 = self.spconv(self.mid_ch, Concatenate(axis=3)([de3_out3, en2_out]), name="msfe3_en_spconv2")
        de3_out1 = self.spconv(self.out_ch, Concatenate(axis=3)([de3_out2, en1_out]), name="msfe3_en_spconv3")

        de_out = self.ctfa(de3_out1, self.out_ch, name_ta='msfe3_en_ta', name_fa='msfe3_en_fa') + en_in

        # Down-sampling
        msfe3_out = self.down_sampling(self.out_ch, de_out, name="msfe3_down_sampling")  # [None, 372, 128 ,64]

        ############################################################################
        # Dilated Dense Block
        ############################################################################
        ddb_out0 = self.dilated_dense_block_in(self.out_ch, msfe3_out, name="ddb_in")
        ddb_out1 = self.dilated_dense_block_1(self.out_ch, ddb_out0, name="ddb_1")
        ddb_out2 = self.dilated_dense_block_2(self.out_ch, Concatenate(axis=3)([ddb_out1, ddb_out0]), name="ddb_2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3(self.out_ch, Concatenate(axis=3)([ddb_out3, ddb_out0]), name="ddb_3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4(self.out_ch, Concatenate(axis=3)([ddb_out4, ddb_out0]), name="ddb_4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5(self.out_ch, Concatenate(axis=3)([ddb_out5, ddb_out0]), name="ddb_5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6(self.out_ch, Concatenate(axis=3)([ddb_out6, ddb_out0]), name="ddb_6")

        ddb_out = self.dilated_dense_block_out(self.out_ch, ddb_out6, name="ddb_out")

        ############################################################################
        # MSFE3 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([ddb_out, msfe3_out]), name="msfe3_upsampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe3_de_in")

        en1_out = self.conv(self.mid_ch, Concatenate(axis=3)([en_in, de3_out1]), name="msfe3_de_conv1")
        en2_out = self.conv(self.mid_ch, Concatenate(axis=3)([en1_out, de3_out2]), name="msfe3_de_conv2")
        en3_out = self.conv(self.mid_ch, Concatenate(axis=3)([en2_out, de3_out3]), name="msfe3_de_conv3")

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in(self.mid_ch, en3_out, name="msfe3_de_ddb_in")
        ddb_out1 = self.dilated_dense_block_1(self.mid_ch, ddb_out0, name="msfe3_de_ddb1")
        ddb_out2 = self.dilated_dense_block_2(self.mid_ch, Concatenate(axis=3)([ddb_out1, ddb_out0]),
                                              name="msfe3_de_ddb2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3(self.mid_ch, Concatenate(axis=3)([ddb_out3, ddb_out0]),
                                              name="msfe3_de_ddb3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4(self.mid_ch, Concatenate(axis=3)([ddb_out4, ddb_out0]),
                                              name="msfe3_de_ddb4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5(self.mid_ch, Concatenate(axis=3)([ddb_out5, ddb_out0]),
                                              name="msfe3_de_ddb5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6(self.mid_ch, Concatenate(axis=3)([ddb_out6, ddb_out0]),
                                              name="msfe3_de_ddb6")

        ddb_out = self.dilated_dense_block_out(self.mid_ch, ddb_out6, name="msfe3_de_ddb_out")

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([ddb_out, en3_out]), name="msfe3_de_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe3_de_spconv2")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe3_de_spconv3")

        de_out = self.ctfa(de_out, self.out_ch, name_ta='msfe3_de_ta', name_fa='msfe3_de_fa') + en_in

        ############################################################################
        # MSFE4 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out3]), name="msfe4_upsampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de_in")
        en1_out = self.conv(self.mid_ch, Concatenate(axis=3)([en_in, de4_3_out1]), name="msfe4_de_conv1")
        en2_out = self.conv(self.mid_ch, Concatenate(axis=3)([en1_out, de4_3_out2]), name="msfe4_de_conv2")
        en3_out = self.conv(self.mid_ch, Concatenate(axis=3)([en2_out, de4_3_out3]), name="msfe4_de_conv3")
        en4_out = self.conv(self.mid_ch, Concatenate(axis=3)([en3_out, de4_3_out4]), name="msfe4_de_conv4")

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in(self.mid_ch, en4_out, name="msfe4_de_ddb_in")
        ddb_out1 = self.dilated_dense_block_1(self.mid_ch, ddb_out0, name="msfe4_de_ddb1")
        ddb_out2 = self.dilated_dense_block_2(self.mid_ch, Concatenate(axis=3)([ddb_out1, ddb_out0]),
                                              name="msfe4_de_ddb2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3(self.mid_ch, Concatenate(axis=3)([ddb_out3, ddb_out0]),
                                              name="msfe4_de_ddb3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4(self.mid_ch, Concatenate(axis=3)([ddb_out4, ddb_out0]),
                                              name="msfe4_de_ddb4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5(self.mid_ch, Concatenate(axis=3)([ddb_out5, ddb_out0]),
                                              name="msfe4_de_ddb5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6(self.mid_ch, Concatenate(axis=3)([ddb_out6, ddb_out0]),
                                              name="msfe4_de_ddb6")

        ddb_out = self.dilated_dense_block_out(self.mid_ch, ddb_out6, name="msfe4_de_ddb_out")

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([ddb_out, en4_out]), name="msfe4_de_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe4_de_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe4_de_spconv3")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe4_de_spconv4")

        de_out = self.ctfa(de_out, self.out_ch, name_ta='msfe4_de_ta', name_fa='msfe4_de_fa') + en_in

        ############################################################################
        # MSFE4(2) - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out2]), name="msfe4_upsampling2")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de2_in")
        en1_out = self.conv(self.mid_ch, Concatenate(axis=3)([en_in, de4_2_out1]), name="msfe4_de2_conv1")
        en2_out = self.conv(self.mid_ch, Concatenate(axis=3)([en1_out, de4_2_out2]), name="msfe4_de2_conv2")
        en3_out = self.conv(self.mid_ch, Concatenate(axis=3)([en2_out, de4_2_out3]), name="msfe4_de2_conv3")
        en4_out = self.conv(self.mid_ch, Concatenate(axis=3)([en3_out, de4_2_out4]), name="msfe4_de2_conv4")

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in(self.mid_ch, en4_out, name="msfe4_de2_ddb_in")
        ddb_out1 = self.dilated_dense_block_1(self.mid_ch, ddb_out0, name="msfe4_de2_ddb1")
        ddb_out2 = self.dilated_dense_block_2(self.mid_ch, Concatenate(axis=3)([ddb_out1, ddb_out0]),
                                              name="msfe4_de2_ddb2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3(self.mid_ch, Concatenate(axis=3)([ddb_out3, ddb_out0]),
                                              name="msfe4_de2_ddb3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4(self.mid_ch, Concatenate(axis=3)([ddb_out4, ddb_out0]),
                                              name="msfe4_de2_ddb4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5(self.mid_ch, Concatenate(axis=3)([ddb_out5, ddb_out0]),
                                              name="msfe4_de2_ddb5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6(self.mid_ch, Concatenate(axis=3)([ddb_out6, ddb_out0]),
                                              name="msfe4_de2_ddb6")

        ddb_out = self.dilated_dense_block_out(self.mid_ch, ddb_out6, name="msfe4_de2_ddb_out")

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([ddb_out, en4_out]), name="msfe4_de2_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe4_de2_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe4_de2_spconv3")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe4_de2_spconv4")

        de_out = self.ctfa(de_out, self.out_ch, name_ta='msfe4_de2_ta', name_fa='msfe4_de2_fa') + en_in

        ############################################################################
        # MSFE4(3) - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out]), name="msfe4_upsampling3")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de3_in")
        en1_out = self.conv(self.mid_ch, Concatenate(axis=3)([en_in, de4_out1]), name="msfe4_de3_conv1")
        en2_out = self.conv(self.mid_ch, Concatenate(axis=3)([en1_out, de4_out2]), name="msfe4_de3_conv2")
        en3_out = self.conv(self.mid_ch, Concatenate(axis=3)([en2_out, de4_out3]), name="msfe4_de3_conv3")
        en4_out = self.conv(self.mid_ch, Concatenate(axis=3)([en3_out, de4_out4]), name="msfe4_de3_conv4")

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in(self.mid_ch, en4_out, name="msfe4_de3_ddb_in")
        ddb_out1 = self.dilated_dense_block_1(self.mid_ch, ddb_out0, name="msfe4_de3_ddb1")
        ddb_out2 = self.dilated_dense_block_2(self.mid_ch, Concatenate(axis=3)([ddb_out1, ddb_out0]),
                                              name="msfe4_de3_ddb2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3(self.mid_ch, Concatenate(axis=3)([ddb_out3, ddb_out0]),
                                              name="msfe4_de3_ddb3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4(self.mid_ch, Concatenate(axis=3)([ddb_out4, ddb_out0]),
                                              name="msfe4_de3_ddb4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5(self.mid_ch, Concatenate(axis=3)([ddb_out5, ddb_out0]),
                                              name="msfe4_de3_ddb5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6(self.mid_ch, Concatenate(axis=3)([ddb_out6, ddb_out0]),
                                              name="msfe4_de3_ddb6")

        ddb_out = self.dilated_dense_block_out(self.mid_ch, ddb_out6, name="msfe4_de3_ddb_out")

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([ddb_out, en4_out]), name="msfe4_de3_spconv1")
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

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in(self.mid_ch, en5_out, name="msfe5_de_ddb_in")
        ddb_out1 = self.dilated_dense_block_1(self.mid_ch, ddb_out0, name="msfe5_de_ddb1")
        ddb_out2 = self.dilated_dense_block_2(self.mid_ch, Concatenate(axis=3)([ddb_out1, ddb_out0]),
                                              name="msfe5_de_ddb2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3(self.mid_ch, Concatenate(axis=3)([ddb_out3, ddb_out0]),
                                              name="msfe5_de_ddb3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4(self.mid_ch, Concatenate(axis=3)([ddb_out4, ddb_out0]),
                                              name="msfe5_de_ddb4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5(self.mid_ch, Concatenate(axis=3)([ddb_out5, ddb_out0]),
                                              name="msfe5_de_ddb5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6(self.mid_ch, Concatenate(axis=3)([ddb_out6, ddb_out0]),
                                              name="msfe5_de_ddb6")

        ddb_out = self.dilated_dense_block_out(self.mid_ch, ddb_out6, name="msfe5_de_ddb_out")

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([ddb_out, en5_out]), name="msfe5_de_spconv1")
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

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in(self.mid_ch, en6_out, name="msfe6_de_ddb_in")
        ddb_out1 = self.dilated_dense_block_1(self.mid_ch, ddb_out0, name="msfe6_de_ddb1")
        ddb_out2 = self.dilated_dense_block_2(self.mid_ch, Concatenate(axis=3)([ddb_out1, ddb_out0]),
                                              name="msfe6_de_ddb2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3(self.mid_ch, Concatenate(axis=3)([ddb_out3, ddb_out0]),
                                              name="msfe6_de_ddb3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4(self.mid_ch, Concatenate(axis=3)([ddb_out4, ddb_out0]),
                                              name="msfe6_de_ddb4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5(self.mid_ch, Concatenate(axis=3)([ddb_out5, ddb_out0]),
                                              name="msfe6_de_ddb5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6(self.mid_ch, Concatenate(axis=3)([ddb_out6, ddb_out0]),
                                              name="msfe6_de_ddb6")

        ddb_out = self.dilated_dense_block_out(self.mid_ch, ddb_out6, name="msfe6_de_ddb_out")

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([ddb_out, en6_out]), name="msfe6_de_spconv1")
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

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(en6_out),
                                                     name="msfe6_en_ddb_in")

        ddb_out1 = self.dilated_dense_block_1_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out0),
                                                    name="msfe6_en_ddb_1")
        ddb_out2 = self.dilated_dense_block_2_valid(self.mid_ch, ZeroPadding2D(padding=((2, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out1, ddb_out0])),
                                                    name="msfe6_en_ddb_2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3_valid(self.mid_ch, ZeroPadding2D(padding=((4, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out3, ddb_out0])),
                                                    name="msfe6_en_ddb_3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4_valid(self.mid_ch, ZeroPadding2D(padding=((8, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out4, ddb_out0])),
                                                    name="msfe6_en_ddb_4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5_valid(self.mid_ch, ZeroPadding2D(padding=((16, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out5, ddb_out0])),
                                                    name="msfe6_en_ddb_5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6_valid(self.mid_ch, ZeroPadding2D(padding=((32, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out6, ddb_out0])),
                                                    name="msfe6_en_ddb_6")

        ddb_out = self.dilated_dense_block_out_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out6),
                                                     name="msfe6_en_ddb_out")

        # Decoder
        de6_out6 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([ddb_out, en6_out])),
                                     name="msfe6_en_spconv1")
        de6_out5 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de6_out6, en5_out])),
                                     name="msfe6_en_spconv2")
        de6_out4 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de6_out5, en4_out])),
                                     name="msfe6_en_spconv3")
        de6_out3 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de6_out4, en3_out])),
                                     name="msfe6_en_spconv4")
        de6_out2 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de6_out3, en2_out])),
                                     name="msfe6_en_spconv5")
        de6_out1 = self.spconv_valid(self.out_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de6_out2, en1_out])),
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

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(en5_out),
                                                     name="msfe5_en_ddb_in")
        ddb_out1 = self.dilated_dense_block_1_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out0),
                                                    name="msfe5_en_ddb_1")
        ddb_out2 = self.dilated_dense_block_2_valid(self.mid_ch, ZeroPadding2D(padding=((2, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out1, ddb_out0])),
                                                    name="msfe5_en_ddb_2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3_valid(self.mid_ch, ZeroPadding2D(padding=((4, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out3, ddb_out0])),
                                                    name="msfe5_en_ddb_3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4_valid(self.mid_ch, ZeroPadding2D(padding=((8, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out4, ddb_out0])),
                                                    name="msfe5_en_ddb_4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5_valid(self.mid_ch, ZeroPadding2D(padding=((16, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out5, ddb_out0])),
                                                    name="msfe5_en_ddb_5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6_valid(self.mid_ch, ZeroPadding2D(padding=((32, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out6, ddb_out0])),
                                                    name="msfe5_en_ddb_6")

        ddb_out = self.dilated_dense_block_out_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out6),
                                                     name="msfe5_en_ddb_out")

        # Decoder
        de5_out5 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([ddb_out, en5_out])),
                                     name="msfe5_en_spconv1")
        de5_out4 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de5_out5, en4_out])),
                                     name="msfe5_en_spconv2")
        de5_out3 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de5_out4, en3_out])),
                                     name="msfe5_en_spconv3")
        de5_out2 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de5_out3, en2_out])),
                                     name="msfe5_en_spconv4")
        de5_out1 = self.spconv_valid(self.out_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de5_out2, en1_out])),
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

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(en4_out),
                                                     name="msfe4_en_ddb_in")
        ddb_out1 = self.dilated_dense_block_1_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out0),
                                                    name="msfe4_en_ddb_1")
        ddb_out2 = self.dilated_dense_block_2_valid(self.mid_ch, ZeroPadding2D(padding=((2, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out1, ddb_out0])),
                                                    name="msfe4_en_ddb_2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3_valid(self.mid_ch, ZeroPadding2D(padding=((4, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out3, ddb_out0])),
                                                    name="msfe4_en_ddb_3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4_valid(self.mid_ch, ZeroPadding2D(padding=((8, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out4, ddb_out0])),
                                                    name="msfe4_en_ddb_4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5_valid(self.mid_ch, ZeroPadding2D(padding=((16, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out5, ddb_out0])),
                                                    name="msfe4_en_ddb_5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6_valid(self.mid_ch, ZeroPadding2D(padding=((32, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out6, ddb_out0])),
                                                    name="msfe4_en_ddb_6")

        ddb_out = self.dilated_dense_block_out_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out6),
                                                     name="msfe4_en_ddb_out")

        # Decoder
        de4_out4 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([ddb_out, en4_out])),
                                     name="msfe4_en_spconv1")
        de4_out3 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de4_out4, en3_out])),
                                     name="msfe4_en_spconv2")
        de4_out2 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de4_out3, en2_out])),
                                     name="msfe4_en_spconv3")
        de4_out1 = self.spconv_valid(self.out_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de4_out2, en1_out])),
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

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(en4_out),
                                                     name="msfe4_en2_ddb_in")
        ddb_out1 = self.dilated_dense_block_1_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out0),
                                                    name="msfe4_en2_ddb_1")
        ddb_out2 = self.dilated_dense_block_2_valid(self.mid_ch, ZeroPadding2D(padding=((2, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out1, ddb_out0])),
                                                    name="msfe4_en2_ddb_2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3_valid(self.mid_ch, ZeroPadding2D(padding=((4, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out3, ddb_out0])),
                                                    name="msfe4_en2_ddb_3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4_valid(self.mid_ch, ZeroPadding2D(padding=((8, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out4, ddb_out0])),
                                                    name="msfe4_en2_ddb_4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5_valid(self.mid_ch, ZeroPadding2D(padding=((16, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out5, ddb_out0])),
                                                    name="msfe4_en2_ddb_5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6_valid(self.mid_ch, ZeroPadding2D(padding=((32, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out6, ddb_out0])),
                                                    name="msfe4_en2_ddb_6")

        ddb_out = self.dilated_dense_block_out_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out6),
                                                     name="msfe4_en2_ddb_out")

        # Decoder
        de4_2_out4 = self.spconv_valid(self.mid_ch,
                                       ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([ddb_out, en4_out])),
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

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(en4_out),
                                                     name="msfe4_en3_ddb_in")
        ddb_out1 = self.dilated_dense_block_1_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out0),
                                                    name="msfe4_en3_ddb_1")
        ddb_out2 = self.dilated_dense_block_2_valid(self.mid_ch, ZeroPadding2D(padding=((2, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out1, ddb_out0])),
                                                    name="msfe4_en3_ddb_2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3_valid(self.mid_ch, ZeroPadding2D(padding=((4, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out3, ddb_out0])),
                                                    name="msfe4_en3_ddb_3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4_valid(self.mid_ch, ZeroPadding2D(padding=((8, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out4, ddb_out0])),
                                                    name="msfe4_en3_ddb_4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5_valid(self.mid_ch, ZeroPadding2D(padding=((16, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out5, ddb_out0])),
                                                    name="msfe4_en3_ddb_5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6_valid(self.mid_ch, ZeroPadding2D(padding=((32, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out6, ddb_out0])),
                                                    name="msfe4_en3_ddb_6")

        ddb_out = self.dilated_dense_block_out_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out6),
                                                     name="msfe4_en3_ddb_out")

        # Decoder
        de4_3_out4 = self.spconv_valid(self.mid_ch,
                                       ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([ddb_out, en4_out])),
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

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(en3_out),
                                                     name="msfe3_en_ddb_in")
        ddb_out1 = self.dilated_dense_block_1_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out0),
                                                    name="msfe3_en_ddb_1")
        ddb_out2 = self.dilated_dense_block_2_valid(self.mid_ch, ZeroPadding2D(padding=((2, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out1, ddb_out0])),
                                                    name="msfe3_en_ddb_2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3_valid(self.mid_ch, ZeroPadding2D(padding=((4, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out3, ddb_out0])),
                                                    name="msfe3_en_ddb_3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4_valid(self.mid_ch, ZeroPadding2D(padding=((8, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out4, ddb_out0])),
                                                    name="msfe3_en_ddb_4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5_valid(self.mid_ch, ZeroPadding2D(padding=((16, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out5, ddb_out0])),
                                                    name="msfe3_en_ddb_5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6_valid(self.mid_ch, ZeroPadding2D(padding=((32, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out6, ddb_out0])),
                                                    name="msfe3_en_ddb_6")

        ddb_out = self.dilated_dense_block_out_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out6),
                                                     name="msfe3_en_ddb_out")

        # Decoder
        de3_out3 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([ddb_out, en3_out])),
                                     name="msfe3_en_spconv1")
        de3_out2 = self.spconv_valid(self.mid_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de3_out3, en2_out])),
                                     name="msfe3_en_spconv2")
        de3_out1 = self.spconv_valid(self.out_ch,
                                     ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([de3_out2, en1_out])),
                                     name="msfe3_en_spconv3")

        de_out = self.ctfa_rt(de3_out1, self.out_ch, name_ta='msfe3_en_ta', name_fa='msfe3_en_fa') + en_in

        # Down-sampling
        msfe3_out = self.down_sampling(self.out_ch, de_out, name="msfe3_down_sampling")  # [None, 372, 128 ,64]

        ############################################################################
        # Dilated Dense Block
        ############################################################################
        ddb_out0 = self.dilated_dense_block_in_valid(self.out_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(msfe3_out),
                                                     name="ddb_in")
        ddb_out1 = self.dilated_dense_block_1_valid(self.out_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out0),
                                                    name="ddb_1")
        ddb_out2 = self.dilated_dense_block_2_valid(self.out_ch, ZeroPadding2D(padding=((2, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out1, ddb_out0])),
                                                    name="ddb_2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3_valid(self.out_ch, ZeroPadding2D(padding=((4, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out3, ddb_out0])),
                                                    name="ddb_3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4_valid(self.out_ch, ZeroPadding2D(padding=((8, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out4, ddb_out0])),
                                                    name="ddb_4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5_valid(self.out_ch, ZeroPadding2D(padding=((16, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out5, ddb_out0])),
                                                    name="ddb_5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6_valid(self.out_ch, ZeroPadding2D(padding=((32, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out6, ddb_out0])),
                                                    name="ddb_6")

        ddb_out = self.dilated_dense_block_out_valid(self.out_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out6),
                                                     name="ddb_out")

        ############################################################################
        # MSFE3 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([ddb_out, msfe3_out]), name="msfe3_upsampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe3_de_in")

        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en_in, de3_out1]))
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe3_de_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en1_out, de3_out2]))
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe3_de_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en2_out, de3_out3]))
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe3_de_conv3")

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(en3_out),
                                                     name="msfe3_de_ddb_in")
        ddb_out1 = self.dilated_dense_block_1_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out0),
                                                    name="msfe3_de_ddb_1")
        ddb_out2 = self.dilated_dense_block_2_valid(self.mid_ch, ZeroPadding2D(padding=((2, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out1, ddb_out0])),
                                                    name="msfe3_de_ddb_2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3_valid(self.mid_ch, ZeroPadding2D(padding=((4, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out3, ddb_out0])),
                                                    name="msfe3_de_ddb_3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4_valid(self.mid_ch, ZeroPadding2D(padding=((8, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out4, ddb_out0])),
                                                    name="msfe3_de_ddb_4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5_valid(self.mid_ch, ZeroPadding2D(padding=((16, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out5, ddb_out0])),
                                                    name="msfe3_de_ddb_5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6_valid(self.mid_ch, ZeroPadding2D(padding=((32, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out6, ddb_out0])),
                                                    name="msfe3_de_ddb_6")

        ddb_out = self.dilated_dense_block_out_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out6),
                                                     name="msfe3_de_ddb_out")

        # Decoder
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([ddb_out, en3_out])),
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
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out3]), name="msfe4_upsampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en_in, de4_3_out1]))
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe4_de_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en1_out, de4_3_out2]))
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe4_de_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en2_out, de4_3_out3]))
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe4_de_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en3_out, de4_3_out4]))
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe4_de_conv4")

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(en4_out),
                                                     name="msfe4_de_ddb_in")
        ddb_out1 = self.dilated_dense_block_1_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out0),
                                                    name="msfe4_de_ddb_1")
        ddb_out2 = self.dilated_dense_block_2_valid(self.mid_ch, ZeroPadding2D(padding=((2, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out1, ddb_out0])),
                                                    name="msfe4_de_ddb_2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3_valid(self.mid_ch, ZeroPadding2D(padding=((4, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out3, ddb_out0])),
                                                    name="msfe4_de_ddb_3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4_valid(self.mid_ch, ZeroPadding2D(padding=((8, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out4, ddb_out0])),
                                                    name="msfe4_de_ddb_4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5_valid(self.mid_ch, ZeroPadding2D(padding=((16, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out5, ddb_out0])),
                                                    name="msfe4_de_ddb_5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6_valid(self.mid_ch, ZeroPadding2D(padding=((32, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out6, ddb_out0])),
                                                    name="msfe4_de_ddb_6")

        ddb_out = self.dilated_dense_block_out_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out6),
                                                     name="msfe4_de_ddb_out")

        # Decoder
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([ddb_out, en4_out])),
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
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out2]), name="msfe4_upsampling2")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de2_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en_in, de4_2_out1]))
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe4_de2_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en1_out, de4_2_out2]))
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe4_de2_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en2_out, de4_2_out3]))
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe4_de2_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en3_out, de4_2_out4]))
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe4_de2_conv4")

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(en4_out),
                                                     name="msfe4_de2_ddb_in")
        ddb_out1 = self.dilated_dense_block_1_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out0),
                                                    name="msfe4_de2_ddb_1")
        ddb_out2 = self.dilated_dense_block_2_valid(self.mid_ch, ZeroPadding2D(padding=((2, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out1, ddb_out0])),
                                                    name="msfe4_de2_ddb_2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3_valid(self.mid_ch, ZeroPadding2D(padding=((4, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out3, ddb_out0])),
                                                    name="msfe4_de2_ddb_3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4_valid(self.mid_ch, ZeroPadding2D(padding=((8, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out4, ddb_out0])),
                                                    name="msfe4_de2_ddb_4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5_valid(self.mid_ch, ZeroPadding2D(padding=((16, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out5, ddb_out0])),
                                                    name="msfe4_de2_ddb_5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6_valid(self.mid_ch, ZeroPadding2D(padding=((32, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out6, ddb_out0])),
                                                    name="msfe4_de2_ddb_6")

        ddb_out = self.dilated_dense_block_out_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out6),
                                                     name="msfe4_de2_ddb_out")

        # Decoder
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([ddb_out, en4_out])),
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
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out]), name="msfe4_upsampling3")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de3_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en_in, de4_out1]))
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe4_de3_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en1_out, de4_out2]))
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe4_de3_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en2_out, de4_out3]))
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe4_de3_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([en3_out, de4_out4]))
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe4_de3_conv4")

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(en4_out),
                                                     name="msfe4_de3_ddb_in")
        ddb_out1 = self.dilated_dense_block_1_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out0),
                                                    name="msfe4_de3_ddb_1")
        ddb_out2 = self.dilated_dense_block_2_valid(self.mid_ch, ZeroPadding2D(padding=((2, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out1, ddb_out0])),
                                                    name="msfe4_de3_ddb_2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3_valid(self.mid_ch, ZeroPadding2D(padding=((4, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out3, ddb_out0])),
                                                    name="msfe4_de3_ddb_3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4_valid(self.mid_ch, ZeroPadding2D(padding=((8, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out4, ddb_out0])),
                                                    name="msfe4_de3_ddb_4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5_valid(self.mid_ch, ZeroPadding2D(padding=((16, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out5, ddb_out0])),
                                                    name="msfe4_de3_ddb_5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6_valid(self.mid_ch, ZeroPadding2D(padding=((32, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out6, ddb_out0])),
                                                    name="msfe4_de3_ddb_6")

        ddb_out = self.dilated_dense_block_out_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out6),
                                                     name="msfe4_de3_ddb_out")

        # Decoder
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([ddb_out, en4_out])),
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

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(en5_out),
                                                     name="msfe5_de_ddb_in")
        ddb_out1 = self.dilated_dense_block_1_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out0),
                                                    name="msfe5_de_ddb_1")
        ddb_out2 = self.dilated_dense_block_2_valid(self.mid_ch, ZeroPadding2D(padding=((2, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out1, ddb_out0])),
                                                    name="msfe5_de_ddb_2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3_valid(self.mid_ch, ZeroPadding2D(padding=((4, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out3, ddb_out0])),
                                                    name="msfe5_de_ddb_3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4_valid(self.mid_ch, ZeroPadding2D(padding=((8, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out4, ddb_out0])),
                                                    name="msfe5_de_ddb_4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5_valid(self.mid_ch, ZeroPadding2D(padding=((16, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out5, ddb_out0])),
                                                    name="msfe5_de_ddb_5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6_valid(self.mid_ch, ZeroPadding2D(padding=((32, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out6, ddb_out0])),
                                                    name="msfe5_de_ddb_6")

        ddb_out = self.dilated_dense_block_out_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out6),
                                                     name="msfe5_de_ddb_out")

        # Decoder
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([ddb_out, en5_out])),
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

        # Dilated Dense Block
        ddb_out0 = self.dilated_dense_block_in_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(en6_out),
                                                     name="msfe6_de_ddb_in")
        ddb_out1 = self.dilated_dense_block_1_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out0),
                                                    name="msfe6_de_ddb_1")
        ddb_out2 = self.dilated_dense_block_2_valid(self.mid_ch, ZeroPadding2D(padding=((2, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out1, ddb_out0])),
                                                    name="msfe6_de_ddb_2")

        ddb_out3 = Concatenate(axis=3)([ddb_out2, ddb_out1])
        ddb_out3 = self.dilated_dense_block_3_valid(self.mid_ch, ZeroPadding2D(padding=((4, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out3, ddb_out0])),
                                                    name="msfe6_de_ddb_3")

        ddb_out4 = Concatenate(axis=3)([ddb_out3, ddb_out2])
        ddb_out4 = Concatenate(axis=3)([ddb_out4, ddb_out1])
        ddb_out4 = self.dilated_dense_block_4_valid(self.mid_ch, ZeroPadding2D(padding=((8, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out4, ddb_out0])),
                                                    name="msfe6_de_ddb_4")

        ddb_out5 = Concatenate(axis=3)([ddb_out4, ddb_out3])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out2])
        ddb_out5 = Concatenate(axis=3)([ddb_out5, ddb_out1])
        ddb_out5 = self.dilated_dense_block_5_valid(self.mid_ch, ZeroPadding2D(padding=((16, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out5, ddb_out0])),
                                                    name="msfe6_de_ddb_5")

        ddb_out6 = Concatenate(axis=3)([ddb_out5, ddb_out4])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out3])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out2])
        ddb_out6 = Concatenate(axis=3)([ddb_out6, ddb_out1])
        ddb_out6 = self.dilated_dense_block_6_valid(self.mid_ch, ZeroPadding2D(padding=((32, 0), (0, 0)))(
            Concatenate(axis=3)([ddb_out6, ddb_out0])),
                                                    name="msfe6_de_ddb_6")

        ddb_out = self.dilated_dense_block_out_valid(self.mid_ch, ZeroPadding2D(padding=((1, 0), (0, 0)))(ddb_out6),
                                                     name="msfe6_de_ddb_out")

        # Decoder
        de_out = self.spconv_valid(self.mid_ch,
                                   ZeroPadding2D(padding=((1, 0), (0, 0)))(Concatenate(axis=3)([ddb_out, en6_out])),
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
