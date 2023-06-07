from keras.layers import Concatenate, Reshape
import tensorflow as tf
import numpy as np
from models import NUTLS_LSTM
import argparse
import options
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

class TFL_SIGNITURE(tf.Module):
    def __init__(self, TFL_MODEL, weight_path):
        self.model = TFL_MODEL
        self.model.load_weights(weight_path)

        for layer in self.model.layers:
            if 'tf' not in layer.name:
                exec(f"self.{layer.name}=layer")

    # Reshape (Before LSTM)
    def reshape_before_lstm(self, x):
        shape = np.shape(x)
        reshape_out = Reshape((shape[1], shape[2] * shape[3]))(x)

        return reshape_out, shape

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, 1, 256, 1], dtype=tf.float32, name='input'),  # Input
                         # Encoder MSFE6 encoder
                         tf.TensorSpec(shape=[None, 1, 256, 64], dtype=tf.float32, name='msfe6_ee_prev1'),
                         tf.TensorSpec(shape=[None, 1, 128, 32], dtype=tf.float32, name='msfe6_ee_prev2'),
                         tf.TensorSpec(shape=[None, 1, 64, 32], dtype=tf.float32, name='msfe6_ee_prev3'),
                         tf.TensorSpec(shape=[None, 1, 32, 32], dtype=tf.float32, name='msfe6_ee_prev4'),
                         tf.TensorSpec(shape=[None, 1, 16, 32], dtype=tf.float32, name='msfe6_ee_prev5'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe6_ee_prev6'),
                         # Encoder MSFE5 encoder
                         tf.TensorSpec(shape=[None, 1, 128, 64], dtype=tf.float32, name='msfe5_ee_prev1'),
                         tf.TensorSpec(shape=[None, 1, 64, 32], dtype=tf.float32, name='msfe5_ee_prev2'),
                         tf.TensorSpec(shape=[None, 1, 32, 32], dtype=tf.float32, name='msfe5_ee_prev3'),
                         tf.TensorSpec(shape=[None, 1, 16, 32], dtype=tf.float32, name='msfe5_ee_prev4'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe5_ee_prev5'),
                         # Encoder MSFE4 encoder
                         tf.TensorSpec(shape=[None, 1, 64, 64], dtype=tf.float32, name='msfe4_ee_prev1'),
                         tf.TensorSpec(shape=[None, 1, 32, 32], dtype=tf.float32, name='msfe4_ee_prev2'),
                         tf.TensorSpec(shape=[None, 1, 16, 32], dtype=tf.float32, name='msfe4_ee_prev3'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe4_ee_prev4'),
                         # Encoder MSFE4 (2) encoder
                         tf.TensorSpec(shape=[None, 1, 32, 64], dtype=tf.float32, name='msfe4_ee2_prev1'),
                         tf.TensorSpec(shape=[None, 1, 16, 32], dtype=tf.float32, name='msfe4_ee2_prev2'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe4_ee2_prev3'),
                         tf.TensorSpec(shape=[None, 1, 4, 32], dtype=tf.float32, name='msfe4_ee2_prev4'),
                         # Encoder MSFE4 (3) encoder
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe4_ee3_prev1'),
                         tf.TensorSpec(shape=[None, 1, 8, 32], dtype=tf.float32, name='msfe4_ee3_prev2'),
                         tf.TensorSpec(shape=[None, 1, 4, 32], dtype=tf.float32, name='msfe4_ee3_prev3'),
                         tf.TensorSpec(shape=[None, 1, 2, 32], dtype=tf.float32, name='msfe4_ee3_prev4'),
                         # Encoder MSFE3 encoder
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe3_ee_prev1'),
                         tf.TensorSpec(shape=[None, 1, 4, 32], dtype=tf.float32, name='msfe3_ee_prev2'),
                         tf.TensorSpec(shape=[None, 1, 2, 32], dtype=tf.float32, name='msfe3_ee_prev3'),

                         # Encoder MSFE6 decoder
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe6_ed_prev1'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe6_ed_prev2'),
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe6_ed_prev3'),
                         tf.TensorSpec(shape=[None, 1, 32, 64], dtype=tf.float32, name='msfe6_ed_prev4'),
                         tf.TensorSpec(shape=[None, 1, 64, 64], dtype=tf.float32, name='msfe6_ed_prev5'),
                         tf.TensorSpec(shape=[None, 1, 128, 64], dtype=tf.float32, name='msfe6_ed_prev6'),
                         # Encoder MSFE5 decoder
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe5_ed_prev1'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe5_ed_prev2'),
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe5_ed_prev3'),
                         tf.TensorSpec(shape=[None, 1, 32, 64], dtype=tf.float32, name='msfe5_ed_prev4'),
                         tf.TensorSpec(shape=[None, 1, 64, 64], dtype=tf.float32, name='msfe5_ed_prev5'),
                         # Encoder MSFE4 decoder
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe4_ed_prev1'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe4_ed_prev2'),
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe4_ed_prev3'),
                         tf.TensorSpec(shape=[None, 1, 32, 64], dtype=tf.float32, name='msfe4_ed_prev4'),
                         # Encoder MSFE4 (2) decoder
                         tf.TensorSpec(shape=[None, 1, 2, 64], dtype=tf.float32, name='msfe4_ed2_prev1'),
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe4_ed2_prev2'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe4_ed2_prev3'),
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe4_ed2_prev4'),
                         # Encoder MSFE4 (3) decoder
                         tf.TensorSpec(shape=[None, 1, 1, 64], dtype=tf.float32, name='msfe4_ed3_prev1'),
                         tf.TensorSpec(shape=[None, 1, 2, 64], dtype=tf.float32, name='msfe4_ed3_prev2'),
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe4_ed3_prev3'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe4_ed3_prev4'),
                         # Encoder MSFE3 decoder
                         tf.TensorSpec(shape=[None, 1, 1, 64], dtype=tf.float32, name='msfe3_ed_prev1'),
                         tf.TensorSpec(shape=[None, 1, 2, 64], dtype=tf.float32, name='msfe3_ed_prev2'),
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe3_ed_prev3'),

                         # Decoder MSFE3 Encoder
                         tf.TensorSpec(shape=[None, 1, 8, 128], dtype=tf.float32, name='msfe3_de_prev1'),
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe3_de_prev2'),
                         tf.TensorSpec(shape=[None, 1, 2, 64], dtype=tf.float32, name='msfe3_de_prev3'),
                         # Decoder MSFE4 Encoder
                         tf.TensorSpec(shape=[None, 1, 16, 128], dtype=tf.float32, name='msfe4_de_prev1'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe4_de_prev2'),
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe4_de_prev3'),
                         tf.TensorSpec(shape=[None, 1, 2, 64], dtype=tf.float32, name='msfe4_de_prev4'),
                         # Decoder MSFE4 (2) Encoder
                         tf.TensorSpec(shape=[None, 1, 32, 128], dtype=tf.float32, name='msfe4_de2_prev1'),
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe4_de2_prev2'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe4_de2_prev3'),
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe4_de2_prev4'),
                         # Decoder MSFE4 (3) Encoder
                         tf.TensorSpec(shape=[None, 1, 64, 128], dtype=tf.float32, name='msfe4_de3_prev1'),
                         tf.TensorSpec(shape=[None, 1, 32, 64], dtype=tf.float32, name='msfe4_de3_prev2'),
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe4_de3_prev3'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe4_de3_prev4'),
                         # Decoder MSFE5 Encoder
                         tf.TensorSpec(shape=[None, 1, 128, 128], dtype=tf.float32, name='msfe5_de_prev1'),
                         tf.TensorSpec(shape=[None, 1, 64, 64], dtype=tf.float32, name='msfe5_de_prev2'),
                         tf.TensorSpec(shape=[None, 1, 32, 64], dtype=tf.float32, name='msfe5_de_prev3'),
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe5_de_prev4'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe5_de_prev5'),
                         # Decoder MSFE6 Encoder
                         tf.TensorSpec(shape=[None, 1, 256, 128], dtype=tf.float32, name='msfe6_de_prev1'),
                         tf.TensorSpec(shape=[None, 1, 128, 64], dtype=tf.float32, name='msfe6_de_prev2'),
                         tf.TensorSpec(shape=[None, 1, 64, 64], dtype=tf.float32, name='msfe6_de_prev3'),
                         tf.TensorSpec(shape=[None, 1, 32, 64], dtype=tf.float32, name='msfe6_de_prev4'),
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe6_de_prev5'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe6_de_prev6'),

                         # Decoder MSFE3 decoder
                         tf.TensorSpec(shape=[None, 1, 1, 64], dtype=tf.float32, name='msfe3_dd_prev1'),
                         tf.TensorSpec(shape=[None, 1, 2, 64], dtype=tf.float32, name='msfe3_dd_prev2'),
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe3_dd_prev3'),
                         # Decoder MSFE4 decoder
                         tf.TensorSpec(shape=[None, 1, 1, 64], dtype=tf.float32, name='msfe4_dd_prev1'),
                         tf.TensorSpec(shape=[None, 1, 2, 64], dtype=tf.float32, name='msfe4_dd_prev2'),
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe4_dd_prev3'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe4_dd_prev4'),
                         # Decoder MSFE4 (2) decoder
                         tf.TensorSpec(shape=[None, 1, 2, 64], dtype=tf.float32, name='msfe4_dd2_prev1'),
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe4_dd2_prev2'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe4_dd2_prev3'),
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe4_dd2_prev4'),
                         # Decoder MSFE4 (3) decoder
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe4_dd3_prev1'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe4_dd3_prev2'),
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe4_dd3_prev3'),
                         tf.TensorSpec(shape=[None, 1, 32, 64], dtype=tf.float32, name='msfe4_dd3_prev4'),
                         # Decoder MSFE5 decoder
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe5_dd_prev1'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe5_dd_prev2'),
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe5_dd_prev3'),
                         tf.TensorSpec(shape=[None, 1, 32, 64], dtype=tf.float32, name='msfe5_dd_prev4'),
                         tf.TensorSpec(shape=[None, 1, 64, 64], dtype=tf.float32, name='msfe5_dd_prev5'),
                         # Decoder MSFE6 decoder
                         tf.TensorSpec(shape=[None, 1, 4, 64], dtype=tf.float32, name='msfe6_dd_prev1'),
                         tf.TensorSpec(shape=[None, 1, 8, 64], dtype=tf.float32, name='msfe6_dd_prev2'),
                         tf.TensorSpec(shape=[None, 1, 16, 64], dtype=tf.float32, name='msfe6_dd_prev3'),
                         tf.TensorSpec(shape=[None, 1, 32, 64], dtype=tf.float32, name='msfe6_dd_prev4'),
                         tf.TensorSpec(shape=[None, 1, 64, 64], dtype=tf.float32, name='msfe6_dd_prev5'),
                         tf.TensorSpec(shape=[None, 1, 128, 64], dtype=tf.float32, name='msfe6_dd_prev6'),

                         # State for LSTM
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe6_en_h'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe6_en_c'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe5_en_h'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe5_en_c'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe4_en_h'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe4_en_c'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe4_en2_h'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe4_en2_c'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe4_en3_h'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe4_en3_c'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe3_en_h'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe3_en_c'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='state_h'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='state_c'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe3_de_h'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe3_de_c'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe4_de_h'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe4_de_c'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe4_de2_h'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe4_de2_c'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe4_de3_h'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe4_de3_c'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe5_de_h'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe5_de_c'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe6_de_h'),
                         tf.TensorSpec(shape=[1, 21], dtype=tf.float32, name='msfe6_de_c'),
                         ])
    def nutls_lstm(self, input,  # order is important
                   msfe6_ee_prev1, msfe6_ee_prev2, msfe6_ee_prev3, msfe6_ee_prev4, msfe6_ee_prev5, msfe6_ee_prev6,
                   msfe5_ee_prev1, msfe5_ee_prev2, msfe5_ee_prev3, msfe5_ee_prev4, msfe5_ee_prev5,
                   msfe4_ee_prev1, msfe4_ee_prev2, msfe4_ee_prev3, msfe4_ee_prev4,
                   msfe4_ee2_prev1, msfe4_ee2_prev2, msfe4_ee2_prev3, msfe4_ee2_prev4,
                   msfe4_ee3_prev1, msfe4_ee3_prev2, msfe4_ee3_prev3, msfe4_ee3_prev4,
                   msfe3_ee_prev1, msfe3_ee_prev2, msfe3_ee_prev3,

                   msfe6_ed_prev1, msfe6_ed_prev2, msfe6_ed_prev3, msfe6_ed_prev4, msfe6_ed_prev5, msfe6_ed_prev6,
                   msfe5_ed_prev1, msfe5_ed_prev2, msfe5_ed_prev3, msfe5_ed_prev4, msfe5_ed_prev5,
                   msfe4_ed_prev1, msfe4_ed_prev2, msfe4_ed_prev3, msfe4_ed_prev4,
                   msfe4_ed2_prev1, msfe4_ed2_prev2, msfe4_ed2_prev3, msfe4_ed2_prev4,
                   msfe4_ed3_prev1, msfe4_ed3_prev2, msfe4_ed3_prev3, msfe4_ed3_prev4,
                   msfe3_ed_prev1, msfe3_ed_prev2, msfe3_ed_prev3,

                   msfe3_de_prev1, msfe3_de_prev2, msfe3_de_prev3,
                   msfe4_de_prev1, msfe4_de_prev2, msfe4_de_prev3, msfe4_de_prev4,
                   msfe4_de2_prev1, msfe4_de2_prev2, msfe4_de2_prev3, msfe4_de2_prev4,
                   msfe4_de3_prev1, msfe4_de3_prev2, msfe4_de3_prev3, msfe4_de3_prev4,
                   msfe5_de_prev1, msfe5_de_prev2, msfe5_de_prev3, msfe5_de_prev4, msfe5_de_prev5,
                   msfe6_de_prev1, msfe6_de_prev2, msfe6_de_prev3, msfe6_de_prev4, msfe6_de_prev5, msfe6_de_prev6,

                   msfe3_dd_prev1, msfe3_dd_prev2, msfe3_dd_prev3,
                   msfe4_dd_prev1, msfe4_dd_prev2, msfe4_dd_prev3, msfe4_dd_prev4,
                   msfe4_dd2_prev1, msfe4_dd2_prev2, msfe4_dd2_prev3, msfe4_dd2_prev4,
                   msfe4_dd3_prev1, msfe4_dd3_prev2, msfe4_dd3_prev3, msfe4_dd3_prev4,
                   msfe5_dd_prev1, msfe5_dd_prev2, msfe5_dd_prev3, msfe5_dd_prev4, msfe5_dd_prev5,
                   msfe6_dd_prev1, msfe6_dd_prev2, msfe6_dd_prev3, msfe6_dd_prev4, msfe6_dd_prev5, msfe6_dd_prev6,

                   msfe6_en_h, msfe6_en_c, msfe5_en_h, msfe5_en_c, msfe4_en_h, msfe4_en_c, msfe4_en2_h, msfe4_en2_c,
                   msfe4_en3_h, msfe4_en3_c, msfe3_en_h, msfe3_en_c, state_h, state_c, msfe3_de_h, msfe3_de_c,
                   msfe4_de_h, msfe4_de_c, msfe4_de2_h, msfe4_de2_c, msfe4_de3_h, msfe4_de3_c, msfe5_de_h,
                   msfe5_de_c, msfe6_de_h, msfe6_de_c):
        in_out = self.input_layer(input)
        ############################################################################
        # MSFE6 - Encoder
        ############################################################################
        msfe6_ee_in_out = self.msfe6_en_in(in_out)
        msfe6_ee_out1 = self.msfe6_en_conv1(Concatenate(axis=1)([msfe6_ee_prev1, msfe6_ee_in_out]))
        msfe6_ee_out2 = self.msfe6_en_conv2(Concatenate(axis=1)([msfe6_ee_prev2, msfe6_ee_out1]))
        msfe6_ee_out3 = self.msfe6_en_conv3(Concatenate(axis=1)([msfe6_ee_prev3, msfe6_ee_out2]))
        msfe6_ee_out4 = self.msfe6_en_conv4(Concatenate(axis=1)([msfe6_ee_prev4, msfe6_ee_out3]))
        msfe6_ee_out5 = self.msfe6_en_conv5(Concatenate(axis=1)([msfe6_ee_prev5, msfe6_ee_out4]))
        msfe6_ee_out6 = self.msfe6_en_conv6(Concatenate(axis=1)([msfe6_ee_prev6, msfe6_ee_out5]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(msfe6_ee_out6)
        en_lstm, msfe6_en_h, msfe6_en_c = self.msfe6_en_lstm(reshape_out, initial_state=[msfe6_en_h, msfe6_en_c])
        en_lstm = self.msfe6_en_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        msfe6_ed_cur1 = Concatenate(axis=3)([en_lstm, msfe6_ee_out6])
        msfe6_ed_out1 = self.msfe6_en_spconv1(Concatenate(axis=1)([msfe6_ed_prev1, msfe6_ed_cur1]))

        msfe6_ed_cur2 = Concatenate(axis=3)([msfe6_ed_out1, msfe6_ee_out5])
        msfe6_ed_out2 = self.msfe6_en_spconv2(Concatenate(axis=1)([msfe6_ed_prev2, msfe6_ed_cur2]))

        msfe6_ed_cur3 = Concatenate(axis=3)([msfe6_ed_out2, msfe6_ee_out4])
        msfe6_ed_out3 = self.msfe6_en_spconv3(Concatenate(axis=1)([msfe6_ed_prev3, msfe6_ed_cur3]))

        msfe6_ed_cur4 = Concatenate(axis=3)([msfe6_ed_out3, msfe6_ee_out3])
        msfe6_ed_out4 = self.msfe6_en_spconv4(Concatenate(axis=1)([msfe6_ed_prev4, msfe6_ed_cur4]))

        msfe6_ed_cur5 = Concatenate(axis=3)([msfe6_ed_out4, msfe6_ee_out2])
        msfe6_ed_out5 = self.msfe6_en_spconv5(Concatenate(axis=1)([msfe6_ed_prev5, msfe6_ed_cur5]))

        msfe6_ed_cur6 = Concatenate(axis=3)([msfe6_ed_out5, msfe6_ee_out1])
        msfe6_ed_out6 = self.msfe6_en_spconv6(Concatenate(axis=1)([msfe6_ed_prev6, msfe6_ed_cur6]))

        TA = self.msfe6_en_ta(msfe6_ed_out6)
        FA = self.msfe6_en_fa(TA)
        TFA = FA * TA

        msfe6_en_ctfa = msfe6_ed_out6 * TFA + msfe6_ee_in_out

        # Down-sampling
        msfe6_out = self.msfe6_down_sampling(msfe6_en_ctfa)

        ############################################################################
        # MSFE5 - Encoder
        ############################################################################
        msfe5_ee_in_out = self.msfe5_en_in(msfe6_out)
        msfe5_ee_out1 = self.msfe5_en_conv1(Concatenate(axis=1)([msfe5_ee_prev1, msfe5_ee_in_out]))
        msfe5_ee_out2 = self.msfe5_en_conv2(Concatenate(axis=1)([msfe5_ee_prev2, msfe5_ee_out1]))
        msfe5_ee_out3 = self.msfe5_en_conv3(Concatenate(axis=1)([msfe5_ee_prev3, msfe5_ee_out2]))
        msfe5_ee_out4 = self.msfe5_en_conv4(Concatenate(axis=1)([msfe5_ee_prev4, msfe5_ee_out3]))
        msfe5_ee_out5 = self.msfe5_en_conv5(Concatenate(axis=1)([msfe5_ee_prev5, msfe5_ee_out4]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(msfe5_ee_out5)
        en_lstm, msfe5_en_h, msfe5_en_c = self.msfe5_en_lstm(reshape_out, initial_state=[msfe5_en_h, msfe5_en_c])
        en_lstm = self.msfe5_en_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        msfe5_ed_cur1 = Concatenate(axis=3)([en_lstm, msfe5_ee_out5])
        msfe5_ed_out1 = self.msfe5_en_spconv1(Concatenate(axis=1)([msfe5_ed_prev1, msfe5_ed_cur1]))

        msfe5_ed_cur2 = Concatenate(axis=3)([msfe5_ed_out1, msfe5_ee_out4])
        msfe5_ed_out2 = self.msfe5_en_spconv2(Concatenate(axis=1)([msfe5_ed_prev2, msfe5_ed_cur2]))

        msfe5_ed_cur3 = Concatenate(axis=3)([msfe5_ed_out2, msfe5_ee_out3])
        msfe5_ed_out3 = self.msfe5_en_spconv3(Concatenate(axis=1)([msfe5_ed_prev3, msfe5_ed_cur3]))

        msfe5_ed_cur4 = Concatenate(axis=3)([msfe5_ed_out3, msfe5_ee_out2])
        msfe5_ed_out4 = self.msfe5_en_spconv4(Concatenate(axis=1)([msfe5_ed_prev4, msfe5_ed_cur4]))

        msfe5_ed_cur5 = Concatenate(axis=3)([msfe5_ed_out4, msfe5_ee_out1])
        msfe5_ed_out5 = self.msfe5_en_spconv5(Concatenate(axis=1)([msfe5_ed_prev5, msfe5_ed_cur5]))

        TA = self.msfe5_en_ta(msfe5_ed_out5)
        FA = self.msfe5_en_fa(TA)
        TFA = FA * TA

        msfe5_en_ctfa = msfe5_ed_out5 * TFA + msfe5_ee_in_out

        # Down-sampling
        msfe5_out = self.msfe5_down_sampling(msfe5_en_ctfa)

        ############################################################################
        # MSFE4 - Encoder
        ############################################################################
        msfe4_ee_in_out = self.msfe4_en_in(msfe5_out)
        msfe4_ee_out1 = self.msfe4_en_conv1(Concatenate(axis=1)([msfe4_ee_prev1, msfe4_ee_in_out]))
        msfe4_ee_out2 = self.msfe4_en_conv2(Concatenate(axis=1)([msfe4_ee_prev2, msfe4_ee_out1]))
        msfe4_ee_out3 = self.msfe4_en_conv3(Concatenate(axis=1)([msfe4_ee_prev3, msfe4_ee_out2]))
        msfe4_ee_out4 = self.msfe4_en_conv4(Concatenate(axis=1)([msfe4_ee_prev4, msfe4_ee_out3]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(msfe4_ee_out4)
        en_lstm, msfe4_en_h, msfe4_en_c = self.msfe4_en_lstm(reshape_out, initial_state=[msfe4_en_h, msfe4_en_c])
        en_lstm = self.msfe4_en_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        msfe4_ed_cur1 = Concatenate(axis=3)([en_lstm, msfe4_ee_out4])
        msfe4_ed_out1 = self.msfe4_en_spconv1(Concatenate(axis=1)([msfe4_ed_prev1, msfe4_ed_cur1]))

        msfe4_ed_cur2 = Concatenate(axis=3)([msfe4_ed_out1, msfe4_ee_out3])
        msfe4_ed_out2 = self.msfe4_en_spconv2(Concatenate(axis=1)([msfe4_ed_prev2, msfe4_ed_cur2]))

        msfe4_ed_cur3 = Concatenate(axis=3)([msfe4_ed_out2, msfe4_ee_out2])
        msfe4_ed_out3 = self.msfe4_en_spconv3(Concatenate(axis=1)([msfe4_ed_prev3, msfe4_ed_cur3]))

        msfe4_ed_cur4 = Concatenate(axis=3)([msfe4_ed_out3, msfe4_ee_out1])
        msfe4_ed_out4 = self.msfe4_en_spconv4(Concatenate(axis=1)([msfe4_ed_prev4, msfe4_ed_cur4]))

        TA = self.msfe4_en_ta(msfe4_ed_out4)
        FA = self.msfe4_en_fa(TA)
        TFA = FA * TA

        msfe4_en_ctfa = msfe4_ed_out4 * TFA + msfe4_ee_in_out

        # Down-sampling
        msfe4_out = self.msfe4_down_sampling(msfe4_en_ctfa)

        ############################################################################
        # MSFE4(2) - Encoder
        ############################################################################
        msfe4_ee2_in_out = self.msfe4_en2_in(msfe4_out)
        msfe4_ee2_out1 = self.msfe4_en2_conv1(Concatenate(axis=1)([msfe4_ee2_prev1, msfe4_ee2_in_out]))
        msfe4_ee2_out2 = self.msfe4_en2_conv2(Concatenate(axis=1)([msfe4_ee2_prev2, msfe4_ee2_out1]))
        msfe4_ee2_out3 = self.msfe4_en2_conv3(Concatenate(axis=1)([msfe4_ee2_prev3, msfe4_ee2_out2]))
        msfe4_ee2_out4 = self.msfe4_en2_conv4(Concatenate(axis=1)([msfe4_ee2_prev4, msfe4_ee2_out3]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(msfe4_ee2_out4)
        en_lstm, msfe4_en2_h, msfe4_en2_c = self.msfe4_en2_lstm(reshape_out,
                                                                initial_state=[msfe4_en2_h, msfe4_en2_c])
        en_lstm = self.msfe4_en2_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        msfe4_ed2_cur1 = Concatenate(axis=3)([en_lstm, msfe4_ee2_out4])
        msfe4_ed2_out1 = self.msfe4_en2_spconv1(Concatenate(axis=1)([msfe4_ed2_prev1, msfe4_ed2_cur1]))

        msfe4_ed2_cur2 = Concatenate(axis=3)([msfe4_ed2_out1, msfe4_ee2_out3])
        msfe4_ed2_out2 = self.msfe4_en2_spconv2(Concatenate(axis=1)([msfe4_ed2_prev2, msfe4_ed2_cur2]))

        msfe4_ed2_cur3 = Concatenate(axis=3)([msfe4_ed2_out2, msfe4_ee2_out2])
        msfe4_ed2_out3 = self.msfe4_en2_spconv3(Concatenate(axis=1)([msfe4_ed2_prev3, msfe4_ed2_cur3]))

        msfe4_ed2_cur4 = Concatenate(axis=3)([msfe4_ed2_out3, msfe4_ee2_out1])
        msfe4_ed2_out4 = self.msfe4_en2_spconv4(Concatenate(axis=1)([msfe4_ed2_prev4, msfe4_ed2_cur4]))

        TA = self.msfe4_en2_ta(msfe4_ed2_out4)
        FA = self.msfe4_en2_fa(TA)
        TFA = FA * TA

        msfe4_en2_ctfa = msfe4_ed2_out4 * TFA + msfe4_ee2_in_out

        # Down-sampling
        msfe4_out2 = self.msfe4_down_sampling2(msfe4_en2_ctfa)

        ############################################################################
        # MSFE4(3) - Encoder
        ############################################################################
        msfe4_ee3_in_out = self.msfe4_en3_in(msfe4_out2)
        msfe4_ee3_out1 = self.msfe4_en3_conv1(Concatenate(axis=1)([msfe4_ee3_prev1, msfe4_ee3_in_out]))
        msfe4_ee3_out2 = self.msfe4_en3_conv2(Concatenate(axis=1)([msfe4_ee3_prev2, msfe4_ee3_out1]))
        msfe4_ee3_out3 = self.msfe4_en3_conv3(Concatenate(axis=1)([msfe4_ee3_prev3, msfe4_ee3_out2]))
        msfe4_ee3_out4 = self.msfe4_en3_conv4(Concatenate(axis=1)([msfe4_ee3_prev4, msfe4_ee3_out3]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(msfe4_ee3_out4)
        en_lstm, msfe4_en3_h, msfe4_en3_c = self.msfe4_en3_lstm(reshape_out,
                                                                initial_state=[msfe4_en3_h, msfe4_en3_c])
        en_lstm = self.msfe4_en3_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        msfe4_ed3_cur1 = Concatenate(axis=3)([en_lstm, msfe4_ee3_out4])
        msfe4_ed3_out1 = self.msfe4_en3_spconv1(Concatenate(axis=1)([msfe4_ed3_prev1, msfe4_ed3_cur1]))

        msfe4_ed3_cur2 = Concatenate(axis=3)([msfe4_ed3_out1, msfe4_ee3_out3])
        msfe4_ed3_out2 = self.msfe4_en3_spconv2(Concatenate(axis=1)([msfe4_ed3_prev2, msfe4_ed3_cur2]))

        msfe4_ed3_cur3 = Concatenate(axis=3)([msfe4_ed3_out2, msfe4_ee3_out2])
        msfe4_ed3_out3 = self.msfe4_en3_spconv3(Concatenate(axis=1)([msfe4_ed3_prev3, msfe4_ed3_cur3]))

        msfe4_ed3_cur4 = Concatenate(axis=3)([msfe4_ed3_out3, msfe4_ee3_out1])
        msfe4_ed3_out4 = self.msfe4_en3_spconv4(Concatenate(axis=1)([msfe4_ed3_prev4, msfe4_ed3_cur4]))

        TA = self.msfe4_en3_ta(msfe4_ed3_out4)
        FA = self.msfe4_en3_fa(TA)
        TFA = FA * TA

        msfe4_en3_ctfa = msfe4_ed3_out4 * TFA + msfe4_ee3_in_out
        # Down-sampling
        msfe4_out3 = self.msfe4_down_sampling3(msfe4_en3_ctfa)

        ############################################################################
        # MSFE3 - Encoder
        ############################################################################
        msfe3_ee_in_out = self.msfe3_en_in(msfe4_out3)
        msfe3_ee_out1 = self.msfe3_en_conv1(Concatenate(axis=1)([msfe3_ee_prev1, msfe3_ee_in_out]))
        msfe3_ee_out2 = self.msfe3_en_conv2(Concatenate(axis=1)([msfe3_ee_prev2, msfe3_ee_out1]))
        msfe3_ee_out3 = self.msfe3_en_conv3(Concatenate(axis=1)([msfe3_ee_prev3, msfe3_ee_out2]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(msfe3_ee_out3)
        en_lstm, msfe3_en_h, msfe3_en_c = self.msfe3_en_lstm(reshape_out, initial_state=[msfe3_en_h, msfe3_en_c])
        en_lstm = self.msfe3_en_dense(en_lstm)
        en_lstm = self.reshape_after_lstm(en_lstm, shape)

        # Decoder
        msfe3_ed_cur1 = Concatenate(axis=3)([en_lstm, msfe3_ee_out3])
        msfe3_ed_out1 = self.msfe3_en_spconv1(Concatenate(axis=1)([msfe3_ed_prev1, msfe3_ed_cur1]))

        msfe3_ed_cur2 = Concatenate(axis=3)([msfe3_ed_out1, msfe3_ee_out2])
        msfe3_ed_out2 = self.msfe3_en_spconv2(Concatenate(axis=1)([msfe3_ed_prev2, msfe3_ed_cur2]))

        msfe3_ed_cur3 = Concatenate(axis=3)([msfe3_ed_out2, msfe3_ee_out1])
        msfe3_ed_out3 = self.msfe3_en_spconv3(Concatenate(axis=1)([msfe3_ed_prev3, msfe3_ed_cur3]))

        TA = self.msfe3_en_ta(msfe3_ed_out3)
        FA = self.msfe3_en_fa(TA)
        TFA = FA * TA

        msfe3_en_ctfa = msfe3_ed_out3 * TFA + msfe3_ee_in_out

        # Down-sampling
        msfe3_out = self.msfe3_down_sampling(msfe3_en_ctfa)  # [None, 372, 128 ,64]

        ############################################################################
        # LSTM
        ############################################################################
        reshape_out, shape = self.reshape_before_lstm(msfe3_out)
        lstm_out, state_h, state_c = self.lstm(reshape_out, initial_state=[state_h, state_c])
        lstm_out = self.dense(lstm_out)
        lstm_out = self.reshape_after_lstm(lstm_out, shape)

        ############################################################################
        # MSFE3 - Decoder
        ############################################################################
        msfe3_de_in_out = self.msfe3_upsampling(Concatenate(axis=3)([lstm_out, msfe3_out]))
        msfe3_de_in_out = self.msfe3_de_in(msfe3_de_in_out)

        msfe3_de_cur1 = Concatenate(axis=3)([msfe3_de_in_out, msfe3_ed_out3])
        msfe3_de_out1 = self.msfe3_de_conv1(Concatenate(axis=1)([msfe3_de_prev1, msfe3_de_cur1]))

        msfe3_de_cur2 = Concatenate(axis=3)([msfe3_de_out1, msfe3_ed_out2])
        msfe3_de_out2 = self.msfe3_de_conv2(Concatenate(axis=1)([msfe3_de_prev2, msfe3_de_cur2]))

        msfe3_de_cur3 = Concatenate(axis=3)([msfe3_de_out2, msfe3_ed_out1])
        msfe3_de_out3 = self.msfe3_de_conv3(Concatenate(axis=1)([msfe3_de_prev3, msfe3_de_cur3]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(msfe3_de_out3)
        de_lstm, msfe3_de_h, msfe3_de_c = self.msfe3_de_lstm(reshape_out, initial_state=[msfe3_de_h, msfe3_de_c])
        de_lstm = self.msfe3_de_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        msfe3_dd_cur1 = Concatenate(axis=3)([de_lstm, msfe3_de_out3])
        msfe3_dd_out1 = self.msfe3_de_spconv1(Concatenate(axis=1)([msfe3_dd_prev1, msfe3_dd_cur1]))

        msfe3_dd_cur2 = Concatenate(axis=3)([msfe3_dd_out1, msfe3_de_out2])
        msfe3_dd_out2 = self.msfe3_de_spconv2(Concatenate(axis=1)([msfe3_dd_prev2, msfe3_dd_cur2]))

        msfe3_dd_cur3 = Concatenate(axis=3)([msfe3_dd_out2, msfe3_de_out1])
        msfe3_dd_out3 = self.msfe3_de_spconv3(Concatenate(axis=1)([msfe3_dd_prev3, msfe3_dd_cur3]))

        TA = self.msfe3_de_ta(msfe3_dd_out3)
        FA = self.msfe3_de_fa(TA)
        TFA = FA * TA

        msfe3_de_ctfa = msfe3_dd_out3 * TFA + msfe3_de_in_out
        ############################################################################
        # MSFE4 - Decoder
        ############################################################################
        msfe4_de_in_out = self.msfe4_upsampling(Concatenate(axis=3)([msfe3_de_ctfa, msfe4_out3]))
        msfe4_de_in_out = self.msfe4_de_in(msfe4_de_in_out)

        msfe4_de_cur1 = Concatenate(axis=3)([msfe4_de_in_out, msfe4_ed3_out4])
        msfe4_de_out1 = self.msfe4_de_conv1(Concatenate(axis=1)([msfe4_de_prev1, msfe4_de_cur1]))

        msfe4_de_cur2 = Concatenate(axis=3)([msfe4_de_out1, msfe4_ed3_out3])
        msfe4_de_out2 = self.msfe4_de_conv2(Concatenate(axis=1)([msfe4_de_prev2, msfe4_de_cur2]))

        msfe4_de_cur3 = Concatenate(axis=3)([msfe4_de_out2, msfe4_ed3_out2])
        msfe4_de_out3 = self.msfe4_de_conv3(Concatenate(axis=1)([msfe4_de_prev3, msfe4_de_cur3]))

        msfe4_de_cur4 = Concatenate(axis=3)([msfe4_de_out3, msfe4_ed3_out1])
        msfe4_de_out4 = self.msfe4_de_conv4(Concatenate(axis=1)([msfe4_de_prev4, msfe4_de_cur4]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(msfe4_de_out4)
        de_lstm, msfe4_de_h, msfe4_de_c = self.msfe4_de_lstm(reshape_out, initial_state=[msfe4_de_h, msfe4_de_c])
        de_lstm = self.msfe4_de_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        msfe4_dd_cur1 = Concatenate(axis=3)([de_lstm, msfe4_de_out4])
        msfe4_dd_out1 = self.msfe4_de_spconv1(Concatenate(axis=1)([msfe4_dd_prev1, msfe4_dd_cur1]))

        msfe4_dd_cur2 = Concatenate(axis=3)([msfe4_dd_out1, msfe4_de_out3])
        msfe4_dd_out2 = self.msfe4_de_spconv2(Concatenate(axis=1)([msfe4_dd_prev2, msfe4_dd_cur2]))

        msfe4_dd_cur3 = Concatenate(axis=3)([msfe4_dd_out2, msfe4_de_out2])
        msfe4_dd_out3 = self.msfe4_de_spconv3(Concatenate(axis=1)([msfe4_dd_prev3, msfe4_dd_cur3]))

        msfe4_dd_cur4 = Concatenate(axis=3)([msfe4_dd_out3, msfe4_de_out1])
        msfe4_dd_out4 = self.msfe4_de_spconv4(Concatenate(axis=1)([msfe4_dd_prev4, msfe4_dd_cur4]))

        TA = self.msfe4_de_ta(msfe4_dd_out4)
        FA = self.msfe4_de_fa(TA)
        TFA = FA * TA

        msfe4_de_ctfa = msfe4_dd_out4 * TFA + msfe4_de_in_out
        ############################################################################
        # MSFE4(2) - Decoder
        ############################################################################
        msfe4_de2_in_out = self.msfe4_upsampling2(Concatenate(axis=3)([msfe4_de_ctfa, msfe4_out2]))
        msfe4_de2_in_out = self.msfe4_de2_in(msfe4_de2_in_out)

        msfe4_de2_cur1 = Concatenate(axis=3)([msfe4_de2_in_out, msfe4_ed2_out4])
        msfe4_de2_out1 = self.msfe4_de2_conv1(Concatenate(axis=1)([msfe4_de2_prev1, msfe4_de2_cur1]))

        msfe4_de2_cur2 = Concatenate(axis=3)([msfe4_de2_out1, msfe4_ed2_out3])
        msfe4_de2_out2 = self.msfe4_de2_conv2(Concatenate(axis=1)([msfe4_de2_prev2, msfe4_de2_cur2]))

        msfe4_de2_cur3 = Concatenate(axis=3)([msfe4_de2_out2, msfe4_ed2_out2])
        msfe4_de2_out3 = self.msfe4_de2_conv3(Concatenate(axis=1)([msfe4_de2_prev3, msfe4_de2_cur3]))

        msfe4_de2_cur4 = Concatenate(axis=3)([msfe4_de2_out3, msfe4_ed2_out1])
        msfe4_de2_out4 = self.msfe4_de2_conv4(Concatenate(axis=1)([msfe4_de2_prev4, msfe4_de2_cur4]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(msfe4_de2_out4)
        de_lstm, msfe4_de2_h, msfe4_de2_c = self.msfe4_de2_lstm(reshape_out,
                                                                initial_state=[msfe4_de2_h, msfe4_de2_c])
        de_lstm = self.msfe4_de2_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        msfe4_dd2_cur1 = Concatenate(axis=3)([de_lstm, msfe4_de2_out4])
        msfe4_dd2_out1 = self.msfe4_de2_spconv1(Concatenate(axis=1)([msfe4_dd2_prev1, msfe4_dd2_cur1]))

        msfe4_dd2_cur2 = Concatenate(axis=3)([msfe4_dd2_out1, msfe4_de2_out3])
        msfe4_dd2_out2 = self.msfe4_de2_spconv2(Concatenate(axis=1)([msfe4_dd2_prev2, msfe4_dd2_cur2]))

        msfe4_dd2_cur3 = Concatenate(axis=3)([msfe4_dd2_out2, msfe4_de2_out2])
        msfe4_dd2_out3 = self.msfe4_de2_spconv3(Concatenate(axis=1)([msfe4_dd2_prev3, msfe4_dd2_cur3]))

        msfe4_dd2_cur4 = Concatenate(axis=3)([msfe4_dd2_out3, msfe4_de2_out1])
        msfe4_dd2_out4 = self.msfe4_de2_spconv4(Concatenate(axis=1)([msfe4_dd2_prev4, msfe4_dd2_cur4]))

        TA = self.msfe4_de2_ta(msfe4_dd2_out4)
        FA = self.msfe4_de2_fa(TA)
        TFA = FA * TA

        msfe4_de2_ctfa = msfe4_dd2_out4 * TFA + msfe4_de2_in_out
        ############################################################################
        # MSFE4(3) - Decoder
        ############################################################################
        msfe4_de3_in_out = self.msfe4_upsampling3(Concatenate(axis=3)([msfe4_de2_ctfa, msfe4_out]))
        msfe4_de3_in_out = self.msfe4_de3_in(msfe4_de3_in_out)
        msfe4_de3_cur1 = Concatenate(axis=3)([msfe4_de3_in_out, msfe4_ed_out4])
        msfe4_de3_out1 = self.msfe4_de3_conv1(Concatenate(axis=1)([msfe4_de3_prev1, msfe4_de3_cur1]))

        msfe4_de3_cur2 = Concatenate(axis=3)([msfe4_de3_out1, msfe4_ed_out3])
        msfe4_de3_out2 = self.msfe4_de3_conv2(Concatenate(axis=1)([msfe4_de3_prev2, msfe4_de3_cur2]))

        msfe4_de3_cur3 = Concatenate(axis=3)([msfe4_de3_out2, msfe4_ed_out2])
        msfe4_de3_out3 = self.msfe4_de3_conv3(Concatenate(axis=1)([msfe4_de3_prev3, msfe4_de3_cur3]))

        msfe4_de3_cur4 = Concatenate(axis=3)([msfe4_de3_out3, msfe4_ed_out1])
        msfe4_de3_out4 = self.msfe4_de3_conv4(Concatenate(axis=1)([msfe4_de3_prev4, msfe4_de3_cur4]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(msfe4_de3_out4)
        de_lstm, msfe4_de3_h, msfe4_de3_c = self.msfe4_de3_lstm(reshape_out,
                                                                initial_state=[msfe4_de3_h, msfe4_de3_c])
        de_lstm = self.msfe4_de3_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        msfe4_dd3_cur1 = Concatenate(axis=3)([de_lstm, msfe4_de3_out4])
        msfe4_dd3_out1 = self.msfe4_de3_spconv1(Concatenate(axis=1)([msfe4_dd3_prev1, msfe4_dd3_cur1]))

        msfe4_dd3_cur2 = Concatenate(axis=3)([msfe4_dd3_out1, msfe4_de3_out3])
        msfe4_dd3_out2 = self.msfe4_de3_spconv2(Concatenate(axis=1)([msfe4_dd3_prev2, msfe4_dd3_cur2]))

        msfe4_dd3_cur3 = Concatenate(axis=3)([msfe4_dd3_out2, msfe4_de3_out2])
        msfe4_dd3_out3 = self.msfe4_de3_spconv3(Concatenate(axis=1)([msfe4_dd3_prev3, msfe4_dd3_cur3]))

        msfe4_dd3_cur4 = Concatenate(axis=3)([msfe4_dd3_out3, msfe4_de3_out1])
        msfe4_dd3_out4 = self.msfe4_de3_spconv4(Concatenate(axis=1)([msfe4_dd3_prev4, msfe4_dd3_cur4]))

        TA = self.msfe4_de3_ta(msfe4_dd3_out4)
        FA = self.msfe4_de3_fa(TA)
        TFA = FA * TA

        msfe4_de3_ctfa = msfe4_dd3_out4 * TFA + msfe4_de3_in_out
        ############################################################################
        # MSFE5 - Decoder
        ############################################################################
        msfe5_de_in_out = self.msfe5_upsampling(Concatenate(axis=3)([msfe4_de3_ctfa, msfe5_out]))
        msfe5_de_in_out = self.msfe5_de_in(msfe5_de_in_out)

        msfe5_de_cur1 = Concatenate(axis=3)([msfe5_de_in_out, msfe5_ed_out5])
        msfe5_de_out1 = self.msfe5_de_conv1(Concatenate(axis=1)([msfe5_de_prev1, msfe5_de_cur1]))

        msfe5_de_cur2 = Concatenate(axis=3)([msfe5_de_out1, msfe5_ed_out4])
        msfe5_de_out2 = self.msfe5_de_conv2(Concatenate(axis=1)([msfe5_de_prev2, msfe5_de_cur2]))

        msfe5_de_cur3 = Concatenate(axis=3)([msfe5_de_out2, msfe5_ed_out3])
        msfe5_de_out3 = self.msfe5_de_conv3(Concatenate(axis=1)([msfe5_de_prev3, msfe5_de_cur3]))

        msfe5_de_cur4 = Concatenate(axis=3)([msfe5_de_out3, msfe5_ed_out2])
        msfe5_de_out4 = self.msfe5_de_conv4(Concatenate(axis=1)([msfe5_de_prev4, msfe5_de_cur4]))

        msfe5_de_cur5 = Concatenate(axis=3)([msfe5_de_out4, msfe5_ed_out1])
        msfe5_de_out5 = self.msfe5_de_conv5(Concatenate(axis=1)([msfe5_de_prev5, msfe5_de_cur5]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(msfe5_de_out5)
        de_lstm, msfe5_de_h, msfe5_de_c = self.msfe5_de_lstm(reshape_out, initial_state=[msfe5_de_h, msfe5_de_c])
        de_lstm = self.msfe5_de_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        msfe5_dd_cur1 = Concatenate(axis=3)([de_lstm, msfe5_de_out5])
        msfe5_dd_out1 = self.msfe5_de_spconv1(Concatenate(axis=1)([msfe5_dd_prev1, msfe5_dd_cur1]))

        msfe5_dd_cur2 = Concatenate(axis=3)([msfe5_dd_out1, msfe5_de_out4])
        msfe5_dd_out2 = self.msfe5_de_spconv2(Concatenate(axis=1)([msfe5_dd_prev2, msfe5_dd_cur2]))

        msfe5_dd_cur3 = Concatenate(axis=3)([msfe5_dd_out2, msfe5_de_out3])
        msfe5_dd_out3 = self.msfe5_de_spconv3(Concatenate(axis=1)([msfe5_dd_prev3, msfe5_dd_cur3]))

        msfe5_dd_cur4 = Concatenate(axis=3)([msfe5_dd_out3, msfe5_de_out2])
        msfe5_dd_out4 = self.msfe5_de_spconv4(Concatenate(axis=1)([msfe5_dd_prev4, msfe5_dd_cur4]))

        msfe5_dd_cur5 = Concatenate(axis=3)([msfe5_dd_out4, msfe5_de_out1])
        msfe5_dd_out5 = self.msfe5_de_spconv5(Concatenate(axis=1)([msfe5_dd_prev5, msfe5_dd_cur5]))

        TA = self.msfe5_de_ta(msfe5_dd_out5)
        FA = self.msfe5_de_fa(TA)
        TFA = FA * TA

        msfe5_de_ctfa = msfe5_dd_out5 * TFA + msfe5_de_in_out
        ############################################################################
        # MSFE6 - Decoder
        ############################################################################
        msfe6_de_in_out = self.msfe6_upsampling(Concatenate(axis=3)([msfe5_de_ctfa, msfe6_out]))
        msfe6_de_in_out = self.msfe6_de_in(msfe6_de_in_out)

        msfe6_de_cur1 = Concatenate(axis=3)([msfe6_de_in_out, msfe6_ed_out6])
        msfe6_de_out1 = self.msfe6_de_conv1(Concatenate(axis=1)([msfe6_de_prev1, msfe6_de_cur1]))

        msfe6_de_cur2 = Concatenate(axis=3)([msfe6_de_out1, msfe6_ed_out5])
        msfe6_de_out2 = self.msfe6_de_conv2(Concatenate(axis=1)([msfe6_de_prev2, msfe6_de_cur2]))

        msfe6_de_cur3 = Concatenate(axis=3)([msfe6_de_out2, msfe6_ed_out4])
        msfe6_de_out3 = self.msfe6_de_conv3(Concatenate(axis=1)([msfe6_de_prev3, msfe6_de_cur3]))

        msfe6_de_cur4 = Concatenate(axis=3)([msfe6_de_out3, msfe6_ed_out3])
        msfe6_de_out4 = self.msfe6_de_conv4(Concatenate(axis=1)([msfe6_de_prev4, msfe6_de_cur4]))

        msfe6_de_cur5 = Concatenate(axis=3)([msfe6_de_out4, msfe6_ed_out2])
        msfe6_de_out5 = self.msfe6_de_conv5(Concatenate(axis=1)([msfe6_de_prev5, msfe6_de_cur5]))

        msfe6_de_cur6 = Concatenate(axis=3)([msfe6_de_out5, msfe6_ed_out1])
        msfe6_de_out6 = self.msfe6_de_conv6(Concatenate(axis=1)([msfe6_de_prev6, msfe6_de_cur6]))

        # lstm
        reshape_out, shape = self.reshape_before_lstm(msfe6_de_out6)
        de_lstm, msfe6_de_h, msfe6_de_c = self.msfe6_de_lstm(reshape_out, initial_state=[msfe6_de_h, msfe6_de_c])
        de_lstm = self.msfe6_de_dense(de_lstm)
        de_lstm = self.reshape_after_lstm(de_lstm, shape)

        # Decoder
        msfe6_dd_cur1 = Concatenate(axis=3)([de_lstm, msfe6_de_out6])
        msfe6_dd_out1 = self.msfe6_de_spconv1(Concatenate(axis=1)([msfe6_dd_prev1, msfe6_dd_cur1]))

        msfe6_dd_cur2 = Concatenate(axis=3)([msfe6_dd_out1, msfe6_de_out5])
        msfe6_dd_out2 = self.msfe6_de_spconv2(Concatenate(axis=1)([msfe6_dd_prev2, msfe6_dd_cur2]))

        msfe6_dd_cur3 = Concatenate(axis=3)([msfe6_dd_out2, msfe6_de_out4])
        msfe6_dd_out3 = self.msfe6_de_spconv3(Concatenate(axis=1)([msfe6_dd_prev3, msfe6_dd_cur3]))

        msfe6_dd_cur4 = Concatenate(axis=3)([msfe6_dd_out3, msfe6_de_out3])
        msfe6_dd_out4 = self.msfe6_de_spconv4(Concatenate(axis=1)([msfe6_dd_prev4, msfe6_dd_cur4]))

        msfe6_dd_cur5 = Concatenate(axis=3)([msfe6_dd_out4, msfe6_de_out2])
        msfe6_dd_out5 = self.msfe6_de_spconv5(Concatenate(axis=1)([msfe6_dd_prev5, msfe6_dd_cur5]))

        msfe6_dd_cur6 = Concatenate(axis=3)([msfe6_dd_out5, msfe6_de_out1])
        msfe6_dd_out6 = self.msfe6_de_spconv6(Concatenate(axis=1)([msfe6_dd_prev6, msfe6_dd_cur6]))

        TA = self.msfe6_de_ta(msfe6_dd_out6)
        FA = self.msfe6_de_fa(TA)
        TFA = FA * TA

        msfe6_de_ctfa = msfe6_dd_out6 * TFA + msfe6_de_in_out

        model_out = self.conv2d(msfe6_de_ctfa)

        return {
            # Encoder - Encoder
            "msfe6_ee_cur1": msfe6_ee_in_out,  # MSFE6ee
            "msfe6_ee_cur2": msfe6_ee_out1,
            "msfe6_ee_cur3": msfe6_ee_out2,
            "msfe6_ee_cur4": msfe6_ee_out3,
            "msfe6_ee_cur5": msfe6_ee_out4,
            "msfe6_ee_cur6": msfe6_ee_out5,
            "msfe5_ee_cur1": msfe5_ee_in_out,  # MSFE5ee
            "msfe5_ee_cur2": msfe5_ee_out1,
            "msfe5_ee_cur3": msfe5_ee_out2,
            "msfe5_ee_cur4": msfe5_ee_out3,
            "msfe5_ee_cur5": msfe5_ee_out4,
            "msfe4_ee_cur1": msfe4_ee_in_out,  # MSFE41ee
            "msfe4_ee_cur2": msfe4_ee_out1,
            "msfe4_ee_cur3": msfe4_ee_out2,
            "msfe4_ee_cur4": msfe4_ee_out3,
            "msfe4_ee2_cur1": msfe4_ee2_in_out,  # msfe42ee
            "msfe4_ee2_cur2": msfe4_ee2_out1,
            "msfe4_ee2_cur3": msfe4_ee2_out2,
            "msfe4_ee2_cur4": msfe4_ee2_out3,
            "msfe4_ee3_cur1": msfe4_ee3_in_out,  # msfe43ee
            "msfe4_ee3_cur2": msfe4_ee3_out1,
            "msfe4_ee3_cur3": msfe4_ee3_out2,
            "msfe4_ee3_cur4": msfe4_ee3_out3,
            "msfe3_ee_cur1": msfe3_ee_in_out,  # msfe3ee
            "msfe3_ee_cur2": msfe3_ee_out1,
            "msfe3_ee_cur3": msfe3_ee_out2,
            # Encoder - Decoder
            "msfe6_ed_cur1": msfe6_ed_cur1,  # msfe6ed
            "msfe6_ed_cur2": msfe6_ed_cur2,
            "msfe6_ed_cur3": msfe6_ed_cur3,
            "msfe6_ed_cur4": msfe6_ed_cur4,
            "msfe6_ed_cur5": msfe6_ed_cur5,
            "msfe6_ed_cur6": msfe6_ed_cur6,
            "msfe5_ed_cur1": msfe5_ed_cur1,  # msfe5ed
            "msfe5_ed_cur2": msfe5_ed_cur2,
            "msfe5_ed_cur3": msfe5_ed_cur3,
            "msfe5_ed_cur4": msfe5_ed_cur4,
            "msfe5_ed_cur5": msfe5_ed_cur5,
            "msfe4_ed_cur1": msfe4_ed_cur1,  # msfe4ed
            "msfe4_ed_cur2": msfe4_ed_cur2,
            "msfe4_ed_cur3": msfe4_ed_cur3,
            "msfe4_ed_cur4": msfe4_ed_cur4,
            "msfe4_ed2_cur1": msfe4_ed2_cur1,  # msfe42ed
            "msfe4_ed2_cur2": msfe4_ed2_cur2,
            "msfe4_ed2_cur3": msfe4_ed2_cur3,
            "msfe4_ed2_cur4": msfe4_ed2_cur4,
            "msfe4_ed3_cur1": msfe4_ed3_cur1,  # msfe43ed
            "msfe4_ed3_cur2": msfe4_ed3_cur2,
            "msfe4_ed3_cur3": msfe4_ed3_cur3,
            "msfe4_ed3_cur4": msfe4_ed3_cur4,
            "msfe3_ed_cur1": msfe3_ed_cur1,  # msfe3ed
            "msfe3_ed_cur2": msfe3_ed_cur2,
            "msfe3_ed_cur3": msfe3_ed_cur3,
            # Decoder - Encoder
            "msfe3_de_cur1": msfe3_de_cur1,  # msfe3de
            "msfe3_de_cur2": msfe3_de_cur2,
            "msfe3_de_cur3": msfe3_de_cur3,
            "msfe4_de_cur1": msfe4_de_cur1,  # MSFE41de
            "msfe4_de_cur2": msfe4_de_cur2,
            "msfe4_de_cur3": msfe4_de_cur3,
            "msfe4_de_cur4": msfe4_de_cur4,
            "msfe4_de2_cur1": msfe4_de2_cur1,  # msfe42de
            "msfe4_de2_cur2": msfe4_de2_cur2,
            "msfe4_de2_cur3": msfe4_de2_cur3,
            "msfe4_de2_cur4": msfe4_de2_cur4,
            "msfe4_de3_cur1": msfe4_de3_cur1,  # msfe43de
            "msfe4_de3_cur2": msfe4_de3_cur2,
            "msfe4_de3_cur3": msfe4_de3_cur3,
            "msfe4_de3_cur4": msfe4_de3_cur4,
            "msfe5_de_cur1": msfe5_de_cur1,  # MSFE5de
            "msfe5_de_cur2": msfe5_de_cur2,
            "msfe5_de_cur3": msfe5_de_cur3,
            "msfe5_de_cur4": msfe5_de_cur4,
            "msfe5_de_cur5": msfe5_de_cur5,
            "msfe6_de_cur1": msfe6_de_cur1,  # MSFE6de
            "msfe6_de_cur2": msfe6_de_cur2,
            "msfe6_de_cur3": msfe6_de_cur3,
            "msfe6_de_cur4": msfe6_de_cur4,
            "msfe6_de_cur5": msfe6_de_cur5,
            "msfe6_de_cur6": msfe6_de_cur6,

            # Decoder - Decoder
            "msfe3_dd_cur1": msfe3_dd_cur1,  # msfe3dd
            "msfe3_dd_cur2": msfe3_dd_cur2,
            "msfe3_dd_cur3": msfe3_dd_cur3,
            "msfe4_dd_cur1": msfe4_dd_cur1,  # msfe4dd
            "msfe4_dd_cur2": msfe4_dd_cur2,
            "msfe4_dd_cur3": msfe4_dd_cur3,
            "msfe4_dd_cur4": msfe4_dd_cur4,
            "msfe4_dd2_cur1": msfe4_dd2_cur1,  # msfe42dd
            "msfe4_dd2_cur2": msfe4_dd2_cur2,
            "msfe4_dd2_cur3": msfe4_dd2_cur3,
            "msfe4_dd2_cur4": msfe4_dd2_cur4,
            "msfe4_dd3_cur1": msfe4_dd3_cur1,  # msfe43dd
            "msfe4_dd3_cur2": msfe4_dd3_cur2,
            "msfe4_dd3_cur3": msfe4_dd3_cur3,
            "msfe4_dd3_cur4": msfe4_dd3_cur4,
            "msfe5_dd_cur1": msfe5_dd_cur1,  # msfe5dd
            "msfe5_dd_cur2": msfe5_dd_cur2,
            "msfe5_dd_cur3": msfe5_dd_cur3,
            "msfe5_dd_cur4": msfe5_dd_cur4,
            "msfe5_dd_cur5": msfe5_dd_cur5,
            "msfe6_dd_cur1": msfe6_dd_cur1,  # msfe6dd
            "msfe6_dd_cur2": msfe6_dd_cur2,
            "msfe6_dd_cur3": msfe6_dd_cur3,
            "msfe6_dd_cur4": msfe6_dd_cur4,
            "msfe6_dd_cur5": msfe6_dd_cur5,
            "msfe6_dd_cur6": msfe6_dd_cur6,

            "msfe6_en_h": msfe6_en_h,
            "msfe6_en_c": msfe6_en_c,
            "msfe5_en_h": msfe5_en_h,
            "msfe5_en_c": msfe5_en_c,
            "msfe4_en_h": msfe4_en_h,
            "msfe4_en_c": msfe4_en_c,
            "msfe4_en2_h": msfe4_en2_h,
            "msfe4_en2_c": msfe4_en2_c,
            "msfe4_en3_h": msfe4_en3_h,
            "msfe4_en3_c": msfe4_en3_c,
            "msfe3_en_h": msfe3_en_h,
            "msfe3_en_c": msfe3_en_c,
            "state_h": state_h,  # LSTM
            "state_c": state_c,
            "msfe3_de_h": msfe3_de_h,
            "msfe3_de_c": msfe3_de_c,
            "msfe4_de_h": msfe4_de_h,
            "msfe4_de_c": msfe4_de_c,
            "msfe4_de2_h": msfe4_de2_h,
            "msfe4_de2_c": msfe4_de2_c,
            "msfe4_de3_h": msfe4_de3_h,
            "msfe4_de3_c": msfe4_de3_c,
            "msfe5_de_h": msfe5_de_h,
            "msfe5_de_c": msfe5_de_c,
            "msfe6_de_h": msfe6_de_h,
            "msfe6_de_c": msfe6_de_c,
            "model_out": model_out,
        }

    # Reshape (After LSTM)

    def reshape_after_lstm(self, x, shape):
        lstm_out = Reshape((shape[1], shape[2], shape[3]))(x)

        return lstm_out


if __name__ == "__main__":
    opt = options.Options().init(argparse.ArgumentParser(description='speech enhancement')).parse_args()
    print(opt)

    saved_model = "../saved_model/nutls_lstm_16ms"

    m = NUTLS_LSTM(opt)
    tflite_model = m.tflite_model()

    model_signature = TFL_SIGNITURE(tflite_model, opt.weight_path)

    tf.saved_model.save(
        model_signature, saved_model,
        signatures={
            'nutls_lstm': model_signature.nutls_lstm.get_concrete_function(),
        }
    )

    # Convert the saved model using TFLiteConverter
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    # Print the signatures from the converted model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    signatures = interpreter.get_signature_list()
    print(signatures)

    with open(opt.tflite_path, 'wb') as f:
        f.write(tflite_model)
