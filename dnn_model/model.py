import keras
from keras import Model, Input
import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, LayerNormalization, PReLU, Lambda, Concatenate, Permute, \
    ZeroPadding2D, LSTM, Dense, Reshape
import numpy as np
from tools import ifft_layer, tf_masking


class NUNET_LSTM():
    def __init__(self):
        self.in_ch = 1
        self.mid_ch = 32
        self.out_ch = 64
        self.win_len = 512
        self.fft_len = 512
        self.stride = 128

        ############################################################################
        # Layer for train model
        ############################################################################
        self.msfe6_en_lstm = LSTM(64, return_sequences=True)
        self.msfe6_en_dense = Dense(128)

        self.msfe5_en_lstm = LSTM(64, return_sequences=True)
        self.msfe5_en_dense = Dense(128)

        self.msfe4_en_lstm_1 = LSTM(64, return_sequences=True)
        self.msfe4_en_dense_1 = Dense(128)

        self.msfe4_en_lstm_2 = LSTM(32, return_sequences=True)
        self.msfe4_en_dense_2 = Dense(64)

        self.msfe4_en_lstm_3 = LSTM(16, return_sequences=True)
        self.msfe4_en_dense_3 = Dense(32)

        self.msfe3_en_lstm = LSTM(16, return_sequences=True)
        self.msfe3_en_dense = Dense(32)

        self.lstm = LSTM(128, return_sequences=True)
        self.dense = Dense(256)

        self.msfe3_de_lstm = LSTM(16, return_sequences=True)
        self.msfe3_de_dense = Dense(32)

        self.msfe4_de_lstm_1 = LSTM(16, return_sequences=True)
        self.msfe4_de_dense_1 = Dense(32)

        self.msfe4_de_lstm_2 = LSTM(32, return_sequences=True)
        self.msfe4_de_dense_2 = Dense(64)

        self.msfe4_de_lstm_3 = LSTM(64, return_sequences=True)
        self.msfe4_de_dense_3 = Dense(128)

        self.msfe5_de_lstm = LSTM(64, return_sequences=True)
        self.msfe5_de_dense = Dense(128)

        self.msfe6_de_lstm = LSTM(64, return_sequences=True)
        self.msfe6_de_dense = Dense(128)

        self.out_conv = Conv2D(self.in_ch, kernel_size=1)

        ############################################################################
        # Layer for TFLite model
        ############################################################################
        self.msfe6_en_lstm_tfl = LSTM(64, return_sequences=True, unroll=True, return_state=True)
        self.msfe6_en_dense_tfl = Dense(128)

        self.msfe5_en_lstm_tfl = LSTM(64, return_sequences=True, unroll=True, return_state=True)
        self.msfe5_en_dense_tfl = Dense(128)

        self.msfe4_en_lstm_1_tfl = LSTM(64, return_sequences=True, unroll=True, return_state=True)
        self.msfe4_en_dense_1_tfl = Dense(128)

        self.msfe4_en_lstm_2_tfl = LSTM(32, return_sequences=True, unroll=True, return_state=True)
        self.msfe4_en_dense_2_tfl = Dense(64)

        self.msfe4_en_lstm_3_tfl = LSTM(16, return_sequences=True, unroll=True, return_state=True)
        self.msfe4_en_dense_3_tfl = Dense(32)

        self.msfe3_en_lstm_tfl = LSTM(16, return_sequences=True, unroll=True, return_state=True)
        self.msfe3_en_dense_tfl = Dense(32)

        self.lstm_tfl = LSTM(128, return_sequences=True, unroll=True, return_state=True)
        self.dense_tfl = Dense(256)

        self.msfe3_de_lstm_tfl = LSTM(16, return_sequences=True, unroll=True, return_state=True)
        self.msfe3_de_dense_tfl = Dense(32)

        self.msfe4_de_lstm_1_tfl = LSTM(16, return_sequences=True, unroll=True, return_state=True)
        self.msfe4_de_dense_1_tfl = Dense(32)

        self.msfe4_de_lstm_2_tfl = LSTM(32, return_sequences=True, unroll=True, return_state=True)
        self.msfe4_de_dense_2_tfl = Dense(64)

        self.msfe4_de_lstm_3_tfl = LSTM(64, return_sequences=True, unroll=True, return_state=True)
        self.msfe4_de_dense_3_tfl = Dense(128)

        self.msfe5_de_lstm_tfl = LSTM(64, return_sequences=True, unroll=True, return_state=True)
        self.msfe5_de_dense_tfl = Dense(128)

        self.msfe6_de_lstm_tfl = LSTM(64, return_sequences=True, unroll=True, return_state=True)
        self.msfe6_de_dense_tfl = Dense(128)

    ############################################################################
    # Tools
    ############################################################################
    def conv(self, out_ch, x, name=None):
        conv2d = keras.Sequential([
            ZeroPadding2D(padding=((1, 0), (1, 1))),
            Conv2D(out_ch, kernel_size=(2, 3), strides=(1, 2), padding='valid'),
            LayerNormalization(epsilon=1e-8),
            PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        ], name=name)

        return conv2d(x)

    # for real-time
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
            Conv2D(out_ch * scale_factor, kernel_size=(1, 3), strides=1, padding='same'),  # [B, T, F, C]
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

    def train_model(self):
        x = Input(batch_shape=(None, 48000))

        frames = tf.signal.stft(x, frame_length=self.win_len, frame_step=self.stride, fft_length=self.fft_len,
                                window_fn=tf.signal.hann_window)
        mags = tf.abs(frames)
        phase = tf.math.angle(frames)

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
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en6_out]), name="msfe6_en_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en5_out]), name="msfe6_en_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en4_out]), name="msfe6_en_spconv3")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe6_en_spconv4")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe6_en_spconv5")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe6_en_spconv6")

        de_out += en_in  # [None, 372, 4 ,32]

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

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en5_out)
        en_lstm = self.msfe5_en_lstm(reshape_out)
        en_lstm = self.msfe5_en_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en5_out]), name="msfe5_en_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en4_out]), name="msfe5_en_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe5_en_spconv3")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe5_en_spconv4")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe5_en_spconv5")

        de_out += en_in  # [None, 372, 4 ,32]

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
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en4_out]), name="msfe4_en_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe4_en_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe4_en_spconv3")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe4_en_spconv4")

        de_out += en_in  # [None, 372, 4 ,32]

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

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        en_lstm = self.msfe4_en_lstm_2(reshape_out)
        en_lstm = self.msfe4_en_dense_2(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en4_out]), name="msfe4_en2_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe4_en2_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe4_en2_spconv3")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe4_en2_spconv4")

        de_out += en_in  # [None, 372, 4 ,32]

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

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        en_lstm = self.msfe4_en_lstm_3(reshape_out)
        en_lstm = self.msfe4_en_dense_3(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en4_out]), name="msfe4_en3_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe4_en3_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe4_en3_spconv3")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe4_en3_spconv4")

        de_out += en_in  # [None, 372, 4 ,32]

        # Down-sampling
        msfe4_out3 = self.down_sampling(self.out_ch, de_out, name="msfe4_down_sampling3")  # [None, 372, 128 ,64]

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
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en3_out]), name="msfe3_en_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe3_en_spconv2")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe3_en_spconv3")

        de_out += en_in  # [None, 372, 4 ,32]

        # Down-sampling
        msfe3_out = self.down_sampling(self.out_ch, de_out, name="msfe3_down_sampling")  # [None, 372, 128 ,64]

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
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([lstm_out, msfe3_out]), name="msfe3_up_sampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe3_de_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe3_de_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe3_de_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe3_de_conv3")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en3_out)
        de_lstm = self.msfe3_de_lstm(reshape_out)
        de_lstm = self.msfe3_de_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_lstm, en3_out]), name="msfe3_de_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe3_de_spconv2")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe3_de_spconv3")

        de_out += en_in

        ############################################################################
        # MSFE4 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out3]), name="msfe4_up_sampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe4_de_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe4_de_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe4_de_conv3")
        en4_out = self.conv(self.mid_ch, en3_out, name="msfe4_de_conv4")

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

        de_out += en_in
        ############################################################################
        # MSFE4(2) - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out2]), name="msfe5_up_sampling2")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de2_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe4_de2_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe4_de2_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe4_de2_conv3")
        en4_out = self.conv(self.mid_ch, en3_out, name="msfe4_de2_conv4")

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

        de_out += en_in

        ############################################################################
        # MSFE4(3) - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out]), name="msfe4_up_sampling3")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de3_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe4_de3_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe4_de3_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe4_de3_conv3")
        en4_out = self.conv(self.mid_ch, en3_out, name="msfe4_de3_conv4")

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

        de_out += en_in

        ############################################################################
        # MSFE5 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe5_out]), name="msfe5_up_sampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe5_de_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe5_de_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe5_de_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe5_de_conv3")
        en4_out = self.conv(self.mid_ch, en3_out, name="msfe5_de_conv4")
        en5_out = self.conv(self.mid_ch, en4_out, name="msfe5_de_conv5")

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

        de_out += en_in

        ############################################################################
        # MSFE6 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe6_out]), name="msfe6_up_sampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe6_de_in")
        en1_out = self.conv(self.mid_ch, en_in, name="msfe6_de_conv1")
        en2_out = self.conv(self.mid_ch, en1_out, name="msfe6_de_conv2")
        en3_out = self.conv(self.mid_ch, en2_out, name="msfe6_de_conv3")
        en4_out = self.conv(self.mid_ch, en3_out, name="msfe6_de_conv4")
        en5_out = self.conv(self.mid_ch, en4_out, name="msfe6_de_conv5")
        en6_out = self.conv(self.mid_ch, en5_out, name="msfe6_de_conv6")

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

        de_out += en_in

        out = self.out_conv(de_out)

        est_mags = tf_masking(out, mags)  # T-F Masking

        # ISTFT
        recons = Lambda(ifft_layer, name='istft')([est_mags, phase])  # [Batch, n_frames, frame_size]
        y = tf.clip_by_value(recons, -1, 1)
        y = tf.squeeze(y)

        self.model = Model(inputs=x, outputs=y)
        self.model.summary()

        return self.model

    def tflite_model(self):
        x = Input(batch_shape=(1, 372, 256, 1))  # [Batch, Time, Frequency, Channel

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
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en6_out]), name="msfe6_en_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en5_out]), name="msfe6_en_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en4_out]), name="msfe6_en_spconv3")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe6_en_spconv4")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe6_en_spconv5")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe6_en_spconv6")

        de_out += en_in  # [None, 372, 4 ,32]

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
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en5_out]), name="msfe5_en_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en4_out]), name="msfe5_en_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe5_en_spconv3")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe5_en_spconv4")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe5_en_spconv5")

        de_out += en_in  # [None, 372, 4 ,32]

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
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en4_out]), name="msfe4_en_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe4_en_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe4_en_spconv3")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe4_en_spconv4")

        de_out += en_in  # [None, 372, 4 ,32]

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
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en4_out]), name="msfe4_en2_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe4_en2_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe4_en2_spconv3")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe4_en2_spconv4")

        de_out += en_in  # [None, 372, 4 ,32]

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
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en4_out]), name="msfe4_en3_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe4_en3_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe4_en3_spconv3")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe4_en3_spconv4")

        de_out += en_in  # [None, 372, 4 ,32]

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
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([en_lstm, en3_out]), name="msfe3_en_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe3_en_spconv2")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe3_en_spconv3")

        de_out += en_in  # [None, 372, 4 ,32]

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
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([lstm_out, msfe3_out]), name="msfe3_up_sampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe3_de_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en_in)
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe3_de_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en1_out)
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe3_de_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en2_out)
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe3_de_conv3")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en3_out)
        de_lstm, _, _ = self.msfe3_de_lstm_tfl(reshape_out)
        de_lstm = self.msfe3_de_dense_tfl(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_lstm, en3_out]), name="msfe3_de_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe3_de_spconv2")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe3_de_spconv3")

        de_out += en_in

        ############################################################################
        # MSFE4 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out3]), name="msfe4_up_sampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en_in)
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe4_de_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en1_out)
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe4_de_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en2_out)
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe4_de_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en3_out)
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe4_de_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        de_lstm, _, _ = self.msfe4_de_lstm_1_tfl(reshape_out)
        de_lstm = self.msfe4_de_dense_1_tfl(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_lstm, en4_out]), name="msfe4_de_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe4_de_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe4_de_spconv3")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe4_de_spconv4")

        de_out += en_in
        ############################################################################
        # MSFE4(2) - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out2]), name="msfe5_up_sampling2")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de2_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en_in)
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe4_de2_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en1_out)
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe4_de2_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en2_out)
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe4_de2_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en3_out)
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe4_de2_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        de_lstm, _, _ = self.msfe4_de_lstm_2_tfl(reshape_out)
        de_lstm = self.msfe4_de_dense_2_tfl(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_lstm, en4_out]), name="msfe4_de2_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe4_de2_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe4_de2_spconv3")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe4_de2_spconv4")

        de_out += en_in

        ############################################################################
        # MSFE4(3) - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe4_out]), name="msfe4_up_sampling3")
        en_in = self.inconv(self.out_ch, en_in, name="msfe4_de3_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en_in)
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe4_de3_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en1_out)
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe4_de3_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en2_out)
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe4_de3_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en3_out)
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe4_de3_conv4")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en4_out)
        de_lstm, _, _ = self.msfe4_de_lstm_3_tfl(reshape_out)
        de_lstm = self.msfe4_de_dense_3_tfl(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_lstm, en4_out]), name="msfe4_de3_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe4_de3_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe4_de3_spconv3")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe4_de3_spconv4")

        de_out += en_in

        ############################################################################
        # MSFE5 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe5_out]), name="msfe5_up_sampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe5_de_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en_in)
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe5_de_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en1_out)
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe5_de_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en2_out)
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe5_de_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en3_out)
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe5_de_conv4")
        en4_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en4_out)
        en5_out = self.conv_valid(self.mid_ch, en4_out_, name="msfe5_de_conv5")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en5_out)
        de_lstm, _, _ = self.msfe5_de_lstm_tfl(reshape_out)
        de_lstm = self.msfe5_de_dense_tfl(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_lstm, en5_out]), name="msfe5_de_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en4_out]), name="msfe5_de_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe5_de_spconv3")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe5_de_spconv4")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe5_de_spconv5")

        de_out += en_in

        ############################################################################
        # MSFE6 - Decoder
        ############################################################################
        en_in = self.up_sampling(self.out_ch * 2, Concatenate(axis=3)([de_out, msfe6_out]), name="msfe6_up_sampling")
        en_in = self.inconv(self.out_ch, en_in, name="msfe6_de_in")
        en_in_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en_in)
        en1_out = self.conv_valid(self.mid_ch, en_in_, name="msfe6_de_conv1")
        en1_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en1_out)
        en2_out = self.conv_valid(self.mid_ch, en1_out_, name="msfe6_de_conv2")
        en2_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en2_out)
        en3_out = self.conv_valid(self.mid_ch, en2_out_, name="msfe6_de_conv3")
        en3_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en3_out)
        en4_out = self.conv_valid(self.mid_ch, en3_out_, name="msfe6_de_conv4")
        en4_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en4_out)
        en5_out = self.conv_valid(self.mid_ch, en4_out_, name="msfe6_de_conv5")
        en5_out_ = ZeroPadding2D(padding=((1, 0), (0, 0)))(en5_out)
        en6_out = self.conv_valid(self.mid_ch, en5_out_, name="msfe6_de_conv6")

        # lstm
        reshape_out, shape = self.reshape_before_lstm(en6_out)
        de_lstm, _, _ = self.msfe6_de_lstm_tfl(reshape_out)
        de_lstm = self.msfe6_de_dense_tfl(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_lstm, en6_out]), name="msfe6_de_spconv1")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en5_out]), name="msfe6_de_spconv2")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en4_out]), name="msfe6_de_spconv3")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en3_out]), name="msfe6_de_spconv4")
        de_out = self.spconv(self.mid_ch, Concatenate(axis=3)([de_out, en2_out]), name="msfe6_de_spconv5")
        de_out = self.spconv(self.out_ch, Concatenate(axis=3)([de_out, en1_out]), name="msfe6_de_spconv6")

        de_out += en_in

        y = self.out_conv(de_out)

        model = Model(inputs=x, outputs=y)
        model.summary()

        return model
